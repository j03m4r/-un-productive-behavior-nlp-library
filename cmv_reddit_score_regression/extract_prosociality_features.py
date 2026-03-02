from main import EnsembleEvaluator
import pandas as pd
import json

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    dataset = load_json_file("./cmv_reddit_score_regression/cmv_2024-01-01T00:00:00-2025-12-31T00:00:00/submissions.json")
    print(f"Loaded {len(dataset)} posts from dataset")

    evaluator = EnsembleEvaluator()
    
    post_features = {}
    skipped_posts = {}
    count = 0
    for post_id, post in dataset.items():
        post_text = f"{post['title']}\n{post['selftext']}"
        if len(post_text) > 20480:
            skipped_posts[post_id] = post
            print(f"Skipping post {post_id} due to length ({len(post_text)} characters)")
            continue
        try:
            post_features[post_id] = evaluator.evaluate_utterance(post_text)
        except Exception as e:
            skipped_posts[post_id] = post
            print(f"Error processing post {post_id}: {e}")
            continue
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} posts")
            write_json_file(post_features, "./cmv_reddit_score_regression/post_features.json")
            write_json_file(skipped_posts, "./cmv_reddit_score_regression/skipped_posts.json")

    write_json_file(post_features, "./cmv_reddit_score_regression/post_features.json")
    write_json_file(skipped_posts, "./cmv_reddit_score_regression/skipped_posts.json")

def write_json_to_csv(json_file_path, csv_file_path):
    data = load_json_file(json_file_path)
    data_flat = {}
    for post_id, features in data.items():
        data_flat[post_id] = {}
        for feature_name, feature_value in features.items():
            if feature_name in set(["Hate Speech", "Toxicity"]):
                data_flat[post_id][feature_name.lower().replace(" ", "_")] = feature_value["score"]
            elif feature_name == "Constructiveness":
                for subfeature_name, subfeature_value in feature_value.items():
                    if subfeature_name == "politeness":
                        for subsubfeature_name, subsubfeature_value in subfeature_value.items():
                            data_flat[post_id][f"p_{subsubfeature_name.lower().replace('.', '_')}"] = subsubfeature_value
                    elif subfeature_name == "argumentative_features":
                        for subsubfeature_name, subsubfeature_value in subfeature_value.items():
                            data_flat[post_id][f"arg_{subsubfeature_name.lower().replace('.', '_')}"] = subsubfeature_value
                    else:
                        data_flat[post_id][f"{subfeature_name.lower()}"] = subfeature_value
            elif feature_name == "Sentiment":
                for subfeature_name, subfeature_value in feature_value.items():
                    data_flat[post_id][f"sent_{subfeature_name.lower()}"] = subfeature_value
            else:
                data_flat[post_id][feature_name] = feature_value
    
    submissions_data = load_json_file("./cmv_reddit_score_regression/cmv_2024-01-01T00:00:00-2025-12-31T00:00:00/submissions.json")
    for post_id, post in submissions_data.items():
        if post_id in data_flat:
            data_flat[post_id]["score"] = post["score"]
            data_flat[post_id]["upvote_ratio"] = post["upvote_ratio"]
        else:
            print(f"Post {post_id} not found in features data, skipping score")

    df = pd.DataFrame.from_dict(data_flat, orient='index')
    df.to_csv(csv_file_path)

if __name__ == "__main__":
    write_json_to_csv("./cmv_reddit_score_regression/post_features.json", "./cmv_reddit_score_regression/post_features.csv")