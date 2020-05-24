from Agent import Agent

if __name__ == "__main__":
    ag1 = Agent()
    ag2 = Agent()
    print("--------------------------------- Q-learning ---------------------------------")
    ag1.Q_values(flag=1)  # flag==1 for Q-learning
    print("----------------------------------- SARSA ------------------------------------")
    ag2.Q_values(flag=0)  # else SARSA