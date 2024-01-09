import csv

with open("3_dataset.csv") as f:
    csv_file = csv.reader(f)
    data = list(csv_file)

    s = data[1][:-1]
    g = [['?' for i in range(len(s))] for j in range(len(s))]

    for i in data:
        if i[-1] == "Yes":
            for j in range(len(s)):
                if i[j] != s[j]:
                    s[j] = '?'
                    g[j][j] = '?'

        elif i[-1] == "No":
            for j in range(len(s)):
                if i[j] != s[j]:
                    g[j][j] = s[j]
                else:
                    g[j][j] = "?"

        print("\nSteps of Candidate Elimination Algorithm", data.index(i) + 1)
        print(s)
        print(g)

    gh = []
    for i in g:
        for j in i:
            if j != '?':
                gh.append(i)
                break

    print("\nFinal specific hypothesis:\n", s)
    print("\nFinal general hypothesis:\n", gh)






# Training data sets.
# -------------trainingexamples.csv----------------
# Sunny,Warm,Normal,Strong,Warm,Same,Yes
# Sunny,Warm,High,Strong,Warm,Same,Yes
# Cloudy,Cold,High,Strong,Warm,Change,No
# Sunny,Warm,High,Strong,Cool,Change,Yes

# OUTPUT

# Step 1 of Candidate Elimination Algorithm
# ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
# [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?'
# , '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?'
# , '?', '?'], ['?', '?', '?', '?', '?', '?']]


# Step 2 of Candidate Elimination Algorithm
# ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
# [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?'
# , '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?'
# , '?', '?'], ['?', '?', '?', '?', '?', '?']]


# Step 3 of Candidate Elimination Algorithm
# ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
# [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['
# ?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '
# ?', '?', '?', '?'], ['?', '?', '?', '?', '?', 'Same']]


# Step 4 of Candidate Elimination Algorithm
# ['Sunny', 'Warm', '?', 'Strong', '?', '?']
# [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['
# ?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '
# ?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

# Final Specific hypothesis:
# ['Sunny', 'Warm', '?', 'Strong', '?', '?']
# Final General hypothesis:
# [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?']]