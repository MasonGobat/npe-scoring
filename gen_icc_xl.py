import pandas as pd

def main():
    df = pd.read_excel("output.xlsx")

    human_judge = df["TOTAL NPE"].copy()
    human_judge = human_judge.reset_index()
    human_judge["judge"] = 1
    human_judge.rename(columns={"index" : "target", "TOTAL NPE" : "rating"}, inplace=True)

    computer_judge = df["test_npe_score"].copy()
    computer_judge = computer_judge.reset_index()
    computer_judge["judge"] = 2
    computer_judge.rename(columns={"index" : "target", "test_npe_score" : "rating"}, inplace=True)

    df = human_judge._append(computer_judge)

    df.to_excel("icc_temp.xlsx")

if __name__ == "__main__":
    main()