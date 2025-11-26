from typing import Dict, List, Optional
import csv
import requests
import time

EARLY_ACCESS_GAMES = [
    {
        "app_id": 1145360,
        "name": "Hades",
        "start_date": 1575158400,
        "end_date": 1598572800,
    }
]


def get_reviews_batch(
    app_id: str,
    cursor: str = "*",
    language: str = "english",
    start_date: int = None,
    end_date: int = None,
    date_range_type: bool = False,
) -> Optional[Dict]:
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "cursor": cursor,
        "filter": "all",
        "language": language,
        "purchase_type": "all",
        "num_per_page": 100,
        "json": 1,
    }

    if date_range_type and start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
        params["date_range_type"] = "include"

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(
                f"Error fetching reviews for app {app_id}: HTTP {response.status_code}"
            )
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error for app {app_id}: {e}")
        return None


def collect_early_access_reviews(
    app_id: str,
    app_name: str,
    start_date: int,
    end_date: int,
    max_reviews: int = 500,
) -> List[Dict]:
    reviews = []
    cursor = "*"

    print(f"Collecting early access reviews for {app_name} (App ID: {app_id})...")

    while len(reviews) < max_reviews:
        response = get_reviews_batch(
            app_id=str(app_id),
            cursor=cursor,
            start_date=start_date,
            end_date=end_date,
            date_range_type=True,
        )

        if not response or not response.get("success"):
            print(
                f"Failed to fetch reviews. Success: {
                    response.get('success') if response else 'No data'
                }"
            )
            break

        batch_reviews = response.get("reviews", [])
        if not batch_reviews:
            print(f"No more reviews available. Total collected: {len(reviews)}")
            break

        for review in batch_reviews:
            written_during_ea = review.get("written_during_early_access", False)

            if not written_during_ea:
                continue

            reviews.append(
                {
                    "app_id": app_id,
                    "app_name": app_name,
                    "review_text": review.get("review", ""),
                    "positive": 1 if review.get("voted_up") else 0,
                    "written_during_early_access": written_during_ea,
                }
            )

            if len(reviews) >= max_reviews:
                break

        print(f"Collected {len(reviews)} early access reviews so far...")

        cursor = response.get("cursor")
        if not cursor:
            print("No more pages available.")
            break

        time.sleep(1)  # Be nice to Steam's servers

    print(f"Total early access reviews collected: {len(reviews)}\n")
    return reviews


def collect_post_release_reviews(
    app_id: str,
    app_name: str,
    max_reviews: int = 500,
) -> List[Dict]:
    reviews = []
    cursor = "*"

    print(f"Collecting post-release reviews for {app_name} (App ID: {app_id})...")

    while len(reviews) < max_reviews:
        response = get_reviews_batch(
            app_id=str(app_id),
            cursor=cursor,
            start_date=None,
            end_date=None,
            date_range_type=False,
        )

        if not response or not response.get("success"):
            print(
                f"Failed to fetch reviews. Success: {
                    response.get('success') if response else 'No response'
                }"
            )
            break

        batch_reviews = response.get("reviews", [])
        if not batch_reviews:
            print(f"No more reviews available. Total collected: {len(reviews)}")
            break

        for review in batch_reviews:
            written_during_ea = review.get("written_during_early_access", False)

            if written_during_ea:
                continue

            reviews.append(
                {
                    "app_id": app_id,
                    "app_name": app_name,
                    "review_text": review.get("review", ""),
                    "positive": 1 if review.get("voted_up") else 0,
                }
            )

            if len(reviews) >= max_reviews:
                break

        print(f"Collected {len(reviews)} post-release reviews so far...")

        cursor = response.get("cursor")
        if not cursor:
            print("No more pages available.")
            break

        time.sleep(1)

    print(f"Total post-release reviews collected: {len(reviews)}\n")
    return reviews


def save_to_csv(reviews: List[Dict], filename: str):
    if not reviews:
        print(f"No reviews to save to {filename}")
        return

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reviews[0].keys())
        writer.writeheader()
        writer.writerows(reviews)

    print(f"Saved {len(reviews)} reviews to {filename}\n")


def main():
    for game in EARLY_ACCESS_GAMES:
        app_id = game["app_id"]
        app_name = game["name"]
        start_date = game.get("start_date")
        end_date = game.get("end_date")

        if start_date and end_date:
            ea_reviews = collect_early_access_reviews(
                app_id=app_id,
                app_name=app_name,
                start_date=start_date,
                end_date=end_date,
                max_reviews=10000,
            )

            if ea_reviews:
                save_to_csv(
                    ea_reviews, f"{app_name.replace(':', '_')}_early_access_reviews.csv"
                )

        post_release_reviews = collect_post_release_reviews(
            app_id=app_id, app_name=app_name, max_reviews=10000
        )

        if post_release_reviews:
            save_to_csv(
                post_release_reviews,
                f"{app_name.replace(':', '_')}_post_release_reviews.csv",
            )


if __name__ == "__main__":
    main()
