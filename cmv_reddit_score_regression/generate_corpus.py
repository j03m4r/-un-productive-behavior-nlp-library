import logging, json, asyncio, os
from datetime import datetime, timedelta, timezone
from BAScraper.BAScraper_async import ArcticShiftAsync

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def date_window(date_str: str):
    incident = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    after  = incident.strftime("%Y-%m-%dT00:00:00")
    before = datetime.strptime("2025-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT00:00:00")
    return after, before


async def main():
    base_save_dir = "./cmv_reddit_score_regression"
    after, before = date_window("2024-01-01")
    save_dir = os.path.join(base_save_dir, f"cmv_{after}-{before}")
    os.makedirs(save_dir, exist_ok=True)

    asa = ArcticShiftAsync(
        log_stream_level="WARNING",
        task_num=5,
        pace_mode="manual",
        sleep_sec=2,
        backoff_sec=10,
        max_retries=5,
        save_dir=save_dir
    )

    subs = await asa.fetch(
        mode="submissions_search",
        subreddit="changemyview",
        after=after,
        before=before,
        limit=0,
        sort="asc",
        file_name="submissions",
    )

    log.info(f"  [cmv] {len(subs)} submissions")


if __name__ == "__main__":
    asyncio.run(main())