import utilities as util
import pandas as pd
from tqdm import tqdm

def get_score(text):
    sents = util.get_sents(text)
    windows = sents[0]
    topics = {"hospital" : 0, "malaria" : 0, "farming" : 0, "school" : 0}
    topics_context = []
    #sim = 0
    #c = 0

    for i in range(len(windows)):
        window = windows[i]

        topic = util.get_single_topic_dependent(window)
        while (topic != None):
            topics[topic] += 1
            #sim += util.get_similarity(sents[1][i])
            #c += 1
            topics_context.append({topic : sents[1][i]})
            window = util.remove_topic(window, topic)
            topic = util.get_single_topic_dependent(window)
            

    return topics, topics_context#, (sim / c if c > 0 else 0)

def main():
    df = pd.read_excel("human_computer_npe_full.xlsx")

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
    #df["avg_similarity"] = scores[2]
    df.to_excel("output.xlsx")

if __name__ == "__main__":
    main()