from Agent import Agent

if __name__ == "__main__":
    ag1 = Agent()
    ag2 = Agent()
    print("--------------------------------- Q-learning ---------------------------------")
    ag1.Q_values(str="Q-learning")
    print("----------------------------------- SARSA ------------------------------------")
    ag2.Q_values(str="SARSA")
