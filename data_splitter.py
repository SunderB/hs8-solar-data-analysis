import numpy as np

customer_data = [[] for i in range(1,101)]
with open("customer_data.csv","r") as f:
    first_line = True
    for line in f:
        # Skip first line
        if (first_line):
            first_line = False
            continue

        id = int(line.split(",")[0])
        customer_data[id].append(line)

for i in range(1,101):
    with open(f"data/customer_data_{i}.csv","w") as f2:
        for line in customer_data[i]:
            f2.write(line)
