"""
quick terminal commands
  python -m game_qa.main                   # full run
  python -m game_qa.main --episodes 5      # quick test
  python -m game_qa.main --headless        # no browser window
  python -m game_qa.main --no-dashboard    # skip web UI
  python -m game_qa.main --warmup-only     # collect frames, train detector, exit
"""

import argparse
import asyncio
import logging
import sys
import time

import numpy as np

from game_qa.config import ( 
    FRAME_CAPTURE_INTERVAL,
    MAX_EPISODE_STEPS,
    NUM_EPISODES,
)
from game_qa.capture.game_driver import GameDriver
from game_qa.agents.game_env import Game2048Env
from game_qa.agents.dqn_agent import DQNAgent
from game_qa.detectors.visual_anomaly import VisualAnomalyDetector, FrameDiffDetector
from game_qa.detectors.log_analyzer import ConsoleLogAnalyzer, PerformanceAnalyzer, BugPrioritizer
from game_qa.reporting.bug_report import BugCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("qa.main")


async def warmup_phase(env: Game2048Env, detector: VisualAnomalyDetector, n_steps: int = 200) -> list[np.ndarray]:
    logger.info("=== WARM-UP: collecting %d normal frames ===", n_steps)
    frames = []
    await env.reset()
    for i in range(n_steps):
        frame = await env.driver.screenshot("warmup")
        frames.append(frame)

        action = np.random.randint(0, 4)
        _, _, done, _ = await env.step(action)
        if done:
            await env.reset()

        if (i + 1) % 50 == 0:
            logger.info("  warm-up %d/%d", i + 1, n_steps)

    logger.info("Collected %d frames. Training autoencoder…", len(frames))
    detector.train(frames)
    return frames


async def run_qa(
    episodes: int,
    headless: bool,
    no_dashboard: bool,
    warmup_only: bool,
):

    # initialise all components                                         
  
    driver = GameDriver(headless=headless, save_frames=True)
    await driver.start()

    env = Game2048Env(driver)
    agent = DQNAgent()
    agent.load()  # load prior checkpoint if present

    visual_detector = VisualAnomalyDetector()
    visual_detector.load()  # load saved autoencoder if present

    frame_diff = FrameDiffDetector()
    log_analyzer = ConsoleLogAnalyzer()
    perf_analyzer = PerformanceAnalyzer()
    prioritizer = BugPrioritizer()
    collector = BugCollector()

    if not no_dashboard:
        from game_qa.reporting.dashboard import start_dashboard, update_run_state
        start_dashboard()
        logger.info("Dashboard: http://127.0.0.1:8765")
    else:
        def update_run_state(**kw): pass  # no-op

    # train autoencoder on normal frames                       
    if not visual_detector.trained:
        await warmup_phase(env, visual_detector, n_steps=150)
    else:
        logger.info("Reusing existing autoencoder.")

    if warmup_only:
        logger.info("Warm-up only — exiting.")
        await driver.close()
        return
    
    # OBSERVE HERE
    # main QA loop                                                       
    logger.info("=== QA RUN: %d episodes, max %d steps each ===", episodes, MAX_EPISODE_STEPS)
    update_run_state(status="running")

    total_bugs = 0
    last_capture_time = 0.0

    for episode in range(1, episodes + 1):
        state = await env.reset()
        episode_reward = 0.0
        loss_history = []

        logger.info("Episode %d/%d  ε=%.3f  coverage=%d",
                    episode, episodes, agent.epsilon, agent.coverage_count)

        for step in range(1, MAX_EPISODE_STEPS + 1):
            # action
            force_explore = (step % 50 == 0)   # periodically force random action for coverage
            action = agent.select_action(state, force_explore=force_explore)
            next_state, reward, done, info = await env.step(action)
            agent.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # train
            loss = agent.optimize()
            if loss:
                loss_history.append(loss)

            agent.decay_epsilon()

            # frame capture & anomaly detection
            now = time.time()
            if now - last_capture_time >= FRAME_CAPTURE_INTERVAL:
                last_capture_time = now
                perf_analyzer.record_frame()

                frame = await driver.screenshot(f"ep{episode}_s{step}")

                # Visual autoencoder
                is_anomaly, mse = visual_detector.detect(frame)

                # Frame diff
                is_diff_anomaly, diff_frac = frame_diff.update(frame)

                if is_anomaly or is_diff_anomaly:
                    board = await driver.get_board_state()
                    recon_img = visual_detector.visualize_reconstruction(frame) if is_anomaly else None
                    priority = prioritizer.score_bug(
                        visual_mse=mse,
                        log_severity=5,
                        frame_diff=diff_frac,
                    )
                    bug = collector.add_visual_anomaly(
                        episode=episode,
                        step=step,
                        score=mse,
                        severity=priority["severity"],
                        composite=priority["composite_score"],
                        screenshot=frame,
                        recon=recon_img,
                        board=board,
                        game_score=info["score"],
                        action=info["action"],
                    )
                    total_bugs += 1
                    logger.warning(
                        "  [ANOMALY] ep=%d step=%d mse=%.4f diff=%.2f → %s (%s)",
                        episode, step, mse, diff_frac, priority["severity"], bug.id
                    )

            # update dashboard state
            if step % 20 == 0:
                perf_sum = perf_analyzer.summary()
                update_run_state(
                    episode=episode,
                    step=step,
                    score=info["score"],
                    epsilon=agent.epsilon,
                    coverage=agent.coverage_count,
                    anomalies_detected=total_bugs,
                    js_errors=len(driver.js_errors),
                    hitches=perf_sum.get("hitches", 0),
                    perf_summary=perf_sum,
                )

            if done:
                break

        # end of episode ->> harvest JS errors
        telemetry = driver.get_telemetry()
        log_analyzer.ingest(telemetry)

        for err in driver.js_errors:
            bug = collector.add_js_error(
                episode=episode,
                step=step,
                error_text=err["text"],
                severity=10,
            )
            total_bugs += 1
            logger.error("  [JS ERROR] %s", err["text"][:120])

        driver.js_errors.clear()

        # performance hitches
        perf_sum = perf_analyzer.summary()
        for hitch in perf_sum.get("hitch_details", []):
            if hitch.get("ratio", 1) > 3:
                collector.add_performance_bug(episode=episode, step=step, hitch=hitch)
                total_bugs += 1

        avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0
        logger.info(
            "Episode %d done  reward=%.1f  avg_loss=%.4f  bugs=%d",
            episode, episode_reward, avg_loss, total_bugs,
        )

    agent.save()

    log_report = log_analyzer.analyze()
    perf_report = perf_analyzer.summary()

    if not no_dashboard:
        update_run_state(
            status="done",
            log_analysis=log_report,
            perf_summary=perf_report,
            anomalies_detected=total_bugs,
        )

    #SEPARATE CLEARLY WITH LOGS, DEBUG LATER
    summary = collector.summary()
    logger.info("")
    logger.info("═══════════════════════════════════════════════════")
    logger.info("QA RUN COMPLETE")
    logger.info("  Total bugs found : %d", summary["total"])
    logger.info("  By severity      : %s", summary["by_severity"])
    logger.info("  By type          : %s", summary["by_type"])
    logger.info("  Agent coverage   : %d unique (state,action) pairs", agent.coverage_count)
    logger.info("  Frame hitches    : %d", perf_report.get("hitches", 0))
    logger.info("  JS errors        : %d", log_report.get("category_counts", {}).get("js_error", 0))
    logger.info("═══════════════════════════════════════════════════")

    if not no_dashboard:
        logger.info("Dashboard still running at http://127.0.0.1:8765  (Ctrl-C to quit)")
        try:
            while True:
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
    else:
        await driver.close()

                                                  

def main():
    parser = argparse.ArgumentParser(description="Automated Game QA for 2048")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes to run")
    parser.add_argument("--headless", action="store_true", help="Run browser headlessly")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard")
    parser.add_argument("--warmup-only", action="store_true", help="Only collect frames and train detector")
    args = parser.parse_args()

    asyncio.run(run_qa(
        episodes=args.episodes,
        headless=args.headless,
        no_dashboard=args.no_dashboard,
        warmup_only=args.warmup_only,
    ))


if __name__ == "__main__":
    main()
