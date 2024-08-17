import utilities as util
import pandas as pd
from tqdm import tqdm

def get_score(text):
    windows = util.get_sents(text)
    topics = {"hospital" : 0, "malaria" : 0, "farming" : 0, "school" : 0}
    topics_context = []

    for window in windows:
        topic = util.get_single_topic_dependent(window)

        if topic != None:
            topics[topic] += 1
            topics_context.append({topic : " ".join(window)})

    return topics, topics_context

def main():
    df = pd.read_excel("human_computer_npe_large.xlsx")

    scores = []
    for ind, row in tqdm(df.iterrows(), total=df.shape[0]):
        scores.append(get_score(row["essay_content"]))

    scores = list(zip(*scores))

    df2 = pd.DataFrame(data=scores[0])

    df["test_npe_hospital"] = df2["hospital"]
    df["test_npe_malaria"] = df2["malaria"]
    df["test_npe_farming"] = df2["farming"]
    df["test_npe_school"] = df2["school"]
    df["test_npe_score"] = [sum([1 for y in x.values() if y > 0]) for x in scores[0]]
    df["test_npe_contexts"] = scores[1]
    df.to_excel("output.xlsx")

if __name__ == "__main__":
    main()