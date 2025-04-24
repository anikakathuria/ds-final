from googleapiclient.discovery import build
import pandas as pd
import time

API_KEY = "AIzaSyAib-9WLNyc0LpxE5A4u8yjEyGH1TgS2ps"  # Replace with your own YouTube Data API v3 key
youtube = build("youtube", "v3", developerKey=API_KEY)

SEARCH_TERMS = ["climate change", "global warming", "climate crisis", "climate action", "carbon emissions"]
MAX_RESULTS_PER_TERM = 500  

def get_video_ids(search_term, max_results):
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        request = youtube.search().list(
            q=search_term,
            part="id",
            type="video",
            maxResults=min(50, max_results - len(video_ids)),
            pageToken=next_page_token
        )
        response = request.execute()

        video_ids += [item["id"]["videoId"] for item in response["items"]]
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(1)
    return video_ids

def get_channel_stats(channel_ids):
    stats = {}
    for i in range(0, len(channel_ids), 50):
        request = youtube.channels().list(
            part="statistics",
            id=",".join(channel_ids[i:i+50])
        )
        response = request.execute()
        for item in response["items"]:
            stats[item["id"]] = {
                "subscriber_count": int(item["statistics"].get("subscriberCount", 0)),
                "video_count": int(item["statistics"].get("videoCount", 0)),
                "view_count_total": int(item["statistics"].get("viewCount", 0))
            }
        time.sleep(1)
    return stats

def get_video_details(video_ids):
    videos_data = []
    all_channel_ids = set()

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(chunk)
        )
        response = request.execute()

        for item in response["items"]:
            snippet = item["snippet"]
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})
            channel_id = snippet["channelId"]
            all_channel_ids.add(channel_id)

            videos_data.append({
                "video_id": item["id"],
                "title": snippet["title"],
                "description": snippet["description"],
                "published_at": snippet["publishedAt"],
                "channel_id": channel_id,
                "channel_title": snippet["channelTitle"],
                "tags": snippet.get("tags", []),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "duration": content.get("duration", ""),
                "category_id": snippet.get("categoryId", ""),
                "live_broadcast": snippet.get("liveBroadcastContent", "")
            })

        time.sleep(1)

    channel_stats = get_channel_stats(list(all_channel_ids))

    # Merge channel stats
    for video in videos_data:
        stats = channel_stats.get(video["channel_id"], {})
        video.update(stats)

    return pd.DataFrame(videos_data)

# Run for all search terms
all_video_ids = set()
for term in SEARCH_TERMS:
    ids = get_video_ids(term, MAX_RESULTS_PER_TERM)
    all_video_ids.update(ids)

# Pull data and save
df = get_video_details(list(all_video_ids))
df.to_csv("climate_youtube_with_channels.csv", index=False)
print("Saved:", df.shape, "records to climate_youtube_with_channels.csv")
