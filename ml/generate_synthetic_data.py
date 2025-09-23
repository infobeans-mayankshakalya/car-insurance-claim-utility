import random
import pandas as pd

makes = ["Toyota", "Honda", "Ford", "BMW"]
models = {"Toyota": ["Corolla", "Camry"], "Honda": ["Civic", "Accord"],
          "Ford": ["Focus", "Fusion"], "BMW": ["320i", "X5"]}
damages = ["dent", "scratch", "broken_glass", "bumper"]

def generate_records(n=500):
    data = []
    for _ in range(n):
        make = random.choice(makes)
        model = random.choice(models[make])
        year = random.randint(2005, 2022)
        damage = random.choice(damages)
        severity = random.choice(["minor", "moderate", "severe"])
        base_cost = random.randint(3000, 10000)
        if severity == "moderate":
            cost = base_cost * 1.5
        elif severity == "severe":
            cost = base_cost * 2
        else:
            cost = base_cost
        data.append({"make": make, "model": model, "year": year,
                     "damage": damage, "severity": severity, "cost": cost})
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_records(1000)
    df.to_csv("synthetic_repairs.csv", index=False)
    print("Generated synthetic_repairs.csv with", len(df), "rows")
