import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# 📥 1. CSV'yi oku
df = pd.read_csv("kiraTahmini_tam_duzenli_UTF8.csv", sep=";")

# 🧹 2. Gereksiz kolonları sil
df = df.drop(columns=["city"])

# 🧠 3. roomCount temizle ("2+1" → 2)
def clean_room_count(room: str):
    try:
        return int(str(room).split("+")[0])
    except:
        return 0

df["roomCount"] = df["roomCount"].apply(clean_room_count)

# 🧠 4. district kolonunu one-hot encode et
district_dummies = pd.get_dummies(df["district"], prefix="district")
df = pd.concat([df.drop(columns=["district"]), district_dummies], axis=1)

# 📊 5. Eğitim verileri
X = df.drop(columns=["rent"])
y = df["rent"]

# 🤖 6. Modeli eğit
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 💾 7. Kaydet
joblib.dump(model, "kira_tahmin_model.pkl")
print("✅ İlçeli model eğitildi ve kaydedildi.")
