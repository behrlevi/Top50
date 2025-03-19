# Szükséges könyvtárak importálása
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import StandardScaler # Az adatok normalizáláshoz/standardizálásához

# Mappa létrehozása a képek tárolásához
if not os.path.exists('statisztika'):
    os.makedirs('statisztika')

# CSV fájl beolvasása
df = pd.read_csv('Top-50-musicality-global.csv')

# Az első néhány sor megjelenítése az adatok megértéséhez
print("Adatkészlet előnézete:")
print(df.head())

# Alapvető információk ellenőrzése
print("\nAdatkészlet információk:")
print(df.info())

# Hiányzó értékek ellenőrzése - adattisztítás
print("\nHiányzó értékek:")
print(df.isnull().sum())

df['Rank'] = df['Unnamed: 0'] + 1 # Az első oszlop tartalmazza a rangsort

# A numerikus értékeket tartalmazó adatkeret másolatának létrehozása
numerical_df = df.select_dtypes(include=['float64', 'int64'])
numerical_df = numerical_df.drop(['Unnamed: 0'], axis=1, errors='ignore')  # Index eldobása, ha létezik

# A numerikus jellemzők listája
features = numerical_df.columns.tolist()
'''
 A rangot kiveszem az elemzendő jellemzők közül mivel ez lesz a hasonlítási alap.
 Kiveszem tovább az Instrumentalitást mivel túlságosan extrém értékeket mutat,
 az Mode-ot, mivel csak két étréket vesz fel (0 vagy 1),
 és TSignature-t mivel ez a legtöbb sorban azonos.
'''
features = [f for f in features if f not in ['Rank', 'Mode', 'TSignature']]

# Numerikus jellemzők eloszlása
print("\nNumerikus jellemzők eloszlásának elemzése:")
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(4, 5, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('statisztika/jellem_eloszlas.png')
plt.close()

# Dobozdiagramok az kiugró értékek azonosításához
plt.figure(figsize=(20, 10))
# Standardizálni kell az adatokat, mert az időtartam és a tempó kiugró értékei torzítják a grafikont
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
sns.boxplot(data=df_scaled)
plt.title('Jellemzők dobozdiagramjai (standardizált)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('statisztika/jellem_doboz.png')
plt.close()

# Korrelációs elemzés
print("\nKorreláció a pozícióval:")
for feature in features:
    correlation = df['Rank'].corr(df[feature])
    print(f"Korreláció a Rank és {feature} között: {correlation:.4f}")

# Korrelációs hőtérkép
plt.figure(figsize=(12, 10))
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korrelációs Hőtérkép')
plt.savefig('statisztika/korr_hoter.png')
plt.close()

# A legnépszerűbb dalok jellemzőinek vizsgálata
print("\nAz első 10 és az utolsó 10 dal összehasonlítása:")
# Rangsor alapú rendezés
df_sorted = df.sort_values('Rank')
top_10 = df_sorted.head(10)
bottom_10 = df_sorted.tail(10)

for feature in features:
    print(f"{feature} - Az első 10 átlaga: {top_10[feature].mean():.2f}, Az utolsó 10 átlaga: {bottom_10[feature].mean():.2f}")

# A rang/népszerűség szórásdiagramja
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rank', y='Popularity', data=df)
plt.title('Rang vs Népszerűség')
plt.savefig('statisztika/pozicio_vs_nepszeruseg.png')
plt.close()

# Összefüggés az energia és a táncolhatóság közt
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Energy', y='Danceability', hue='Popularity',
                sizes=(20, 100), data=df)
plt.title('Energia vs Táncolhatóság (színezve és méretezve a népszerűség szerint)')
plt.savefig('statisztika/energia_tancolhatosag.png')
plt.close()

# A népszerűség és más jellemzők közötti kapcsolat
popular_features = ['Danceability', 'Energy', 'Acousticness']
plt.figure(figsize=(15, 5))
for i, feature in enumerate(popular_features):
    if feature in df.columns:
        plt.subplot(1, 3, i + 1)
        sns.scatterplot(x='Popularity', y=feature, data=df)
        plt.title(f'Popularity vs {feature}')

        # Trendvonal hozzáadása
        z = np.polyfit(df['Popularity'], df[feature], 1)
        p = np.poly1d(z)
        plt.plot(df['Popularity'], p(df['Popularity']), "r--")
plt.tight_layout()
plt.savefig('statisztika/nepszeruseg_vs_jellemzok.png')
plt.close()

# Az öt legjobb előadó (a legtöbb toplistás dal előadója -> legsikeresebb)
artist_counts = df['Artist Name'].value_counts()
top_artists = artist_counts.head(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_artists.values, y=top_artists.index)
plt.title('Top 5 előadó a dalok száma szerint')
plt.savefig('statisztika/top_eloadok.png')
plt.close()

# Következtetések
print("\nFő megállapítások összefoglalása:")
correlations = [(feature, abs(df['Rank'].corr(df[feature]))) for feature in features]
correlations.sort(key=lambda x: x[1], reverse=True)
print("A top 3 jellemző, amely a legerősebb összefüggést mutatta a ranggal:")
for feature, corr in correlations[:3]:  # Csak a top 3 jellemzőt mutatjuk
    print(f"   - {feature}: {corr:.4f}")

print("\n2. A vezető és sereghajtó daloknál ezek a tulajdonságok mutatták a legnagyobb eltérést:")
top_features = []
for feature in features:
    # Kihagyom a Key, Duration tulajdonságokat mert érzem őket mérvadónak.
    if feature not in ['Key', 'duration']:
        diff = top_10[feature].mean() - bottom_10[feature].mean()
        if abs(diff) > 0.1:  # Kizárja a jelentéktelen különbségeket
            top_features.append((feature, diff))
            print(f"   - {feature}: Különbség: {diff:.2f}")

# Kiválasztja a három legnagyobb különbséget
top_features.sort(key=lambda x: abs(x[1]), reverse=True)
top_diff_features = [f[0] for f in top_features[:3]]

# A legfőbb különbségek vizualizációja
if top_diff_features:
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(top_diff_features):
        plt.subplot(1, len(top_diff_features), i + 1)
        sns.boxplot(x=['Első 10' if x < 10 else 'Utolsó 10' if x > len(df) - 10 else 'Közép'
                      for x in df_sorted.index], y=feature, data=df_sorted)
        plt.title(f'{feature} értékek összehasonlítása')
    plt.tight_layout()
    plt.savefig('statisztika/legjel_kul.png')
    plt.close()