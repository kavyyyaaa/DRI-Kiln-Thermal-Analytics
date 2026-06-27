import pandas as pd

df = pd.read_excel(
    r"C:\Users\DELL\Downloads\kiln_patch_10day_avg (6).xlsx"
)

print(df.columns.tolist())
print(df.shape)