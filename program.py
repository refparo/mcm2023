# %%
from geopy import distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
import time

STATES = [
  # 'AL', # no data
  # 'AK', # no history
  'AZ', # good
  # 'AR', # no data
  # 'CA', # no history
  'CO', # good
  # 'CT', # no data
  # 'DE', # no data
  # 'FL', # no data
  # 'GA', # no data
  # 'HI', # no history
  'ID', # good
  # 'IL', # no data
  # 'IN', # no data
  # 'IA', # no data
  # 'KS', # no data
  # 'KY', # no data
  # 'LA', # no data
  # 'ME', # no data
  # 'MD', # no data
  # 'MA', # no data
  # 'MI', # no data
  # 'MN', # no data
  # 'MS', # no data
  # 'MO', # no data
  'MT', # good
  # 'NE', # no data
  # 'NV', # no match
  # 'NH', # no data
  # 'NJ', # no data
  # 'NM', # no history
  # 'NY', # no data
  # 'NC', # no data
  # 'ND', # no data
  # 'OH', # no data
  # 'OK', # no data
  # 'OR', # no history
  # 'PA', # no data
  # 'RI', # no data
  # 'SC', # no data
  # 'SD', # no data
  # 'TN', # no data
  # 'TX', # no data
  # 'UT', # no match
  # 'VT', # no data
  # 'VA', # no data
  # 'WA', # no history
  # 'WV', # no data
  # 'WI', # no data
  # 'WY', # no history
]

# %%
plots = pd.concat(pd.read_csv(f'FIADB/{state}_PLOT.csv') for state in STATES)

plots = plots[
  (plots['PLOT_STATUS_CD'] == 1) &
  (plots['P2VEG_SAMPLING_STATUS_CD'] > 0)
]

plots['MEASDATE'] = pd.to_datetime(
  plots['MEASYEAR'].astype('str') + '-' +
  plots['MEASMON'].astype('str').str.zfill(2) + '-' +
  plots['MEASDAY'].astype('str').str.zfill(2)
)

# %%
species = pd.concat(pd.read_csv(f'FIADB/{state}_P2VEG_SUBPLOT_SPP.csv') for state in STATES)
species_dedup = species.drop_duplicates(subset=['PLT_CN', 'VEG_SPCD', 'UNIQUE_SP_NBR'])
richness = species_dedup.groupby(by=['PLT_CN']).size()

plots['RICHNESS'] = plots['CN'].apply(lambda cn:
  richness[cn] if cn in richness else 0)

# %%
plot_groups = plots.groupby(by=['STATECD', 'UNITCD', 'COUNTYCD', 'PLOT'])
plot_group_sizes = plot_groups.size()

def extract_group_data(grp: tuple):
  group = plot_groups.get_group(grp)
  return pd.Series({
    'size': len(group),
    'pos': group.iloc[0][['LAT', 'LON']],
    'start': group['MEASYEAR'].min()
  })

qualified_groups = plot_group_sizes[plot_group_sizes > 2].index.to_series() \
  .apply(extract_group_data)

# %%
sites = pd.read_table('SCAN/SCAN_sites.txt')
sites['pos'] = sites.apply(
  lambda station: tuple(station[['lat', 'lon']]),
  axis=1
)

# %%
def find_site(pos1: tuple[float, float]):
  dist = sites['pos'].apply(lambda pos2: distance.geodesic(pos1, pos2).km)
  site_id = dist.idxmin()
  return pd.Series({
    'site_id': site_id,
    'site_name': sites.iloc[site_id]['site_name'],
    'site_start': sites.iloc[site_id]['start'],
    'site_dist': dist[site_id]
  })

qualified_groups = qualified_groups.assign(
  **qualified_groups['pos'].apply(find_site)
)
qualified_groups = qualified_groups[
  (
    qualified_groups['start'] >
    qualified_groups['site_start'].str.slice(0, 4).astype('int')
  ) &
  (qualified_groups['site_dist'] < 200)
].sort_values(['site_dist'])

# %%
soil_data = {
  site: pd.read_csv(f'SCAN/{site}.csv', comment='#')
  for site in set(qualified_groups['site_name'])
}

MOISTURE_COL = 'Soil Moisture Percent -40in (pct) Mean of Hourly Values'

for site_data in soil_data.values():
  site_data['Date'] = pd.to_datetime(site_data['Date'])

# %%
rng = np.random.Generator(np.random.PCG64(seed=42))

def check_site_early_data(row: pd.Series):
  group = plot_groups.get_group(row.name)
  site_data = soil_data[row['site_name']]
  first_date = group['MEASDATE'].min()
  last_date = group['MEASDATE'].max()
  if np.isnan(site_data.set_index('Date')[MOISTURE_COL][first_date]):
    return False
  moisture = site_data[MOISTURE_COL]
  missing_len = 0
  in_missing = False
  for i in site_data.index[
    site_data['Date'].between(first_date, last_date, inclusive='both')
  ]:
    if in_missing:
      if np.isnan(moisture[i]):
        missing_len += 1
      else:
        if missing_len > 100:
          return False # too long missing interval, drop the data
        # fill in the missing data
        moisture[i - missing_len:i] = (
          np.linspace(
            moisture[i - missing_len - 1],
            moisture[i], missing_len + 2
          )[1:-1] +
          rng.standard_normal(missing_len) / 100
        )
        in_missing = False
    else:
      if np.isnan(moisture[i]):
        in_missing = True
        missing_len = 1
  return True

qualified_groups = qualified_groups[
  qualified_groups.apply(check_site_early_data, axis=1)
]

# %%
def fourier_fitting(y: np.ndarray, harmonics: int = 20):
  n = y.size
  t = np.arange(n)
  k = np.polyfit(t, y, 1)[0]
  y_detrended = y - k * t
  y_freqdom = np.fft.fft(y_detrended)
  y_freq = np.fft.fftfreq(n)
  idx_sorted = sorted(
    range(n),
    key=lambda i: np.absolute(y_freqdom[i]),
    reverse=True
  )

  return lambda t: sum(
    np.absolute(y_freqdom[i]) / n * np.cos(
      2 * np.pi * y_freq[i] * t + np.angle(y_freqdom[i])
    )
    for i in idx_sorted[:1 + 2 * harmonics]
  ) + k * t

# %%
for grp in qualified_groups.index[1], :
  row = qualified_groups.loc[grp]
  group = plot_groups.get_group(grp).sort_values(['MEASYEAR'])
  start = group.iloc[0]['MEASDATE']
  end = group.iloc[-1]['MEASDATE']
  site_data = soil_data[row['site_name']]
  site_data = site_data[site_data['Date'].between(start, end, inclusive='both')]
  plt.plot(site_data.index, site_data[MOISTURE_COL])
  #group.to_csv(f'output/vegetation_data_{grp[3]}.csv', index=False)
  #site_data.to_csv(f'output/soil_data_{grp[3]}.csv', index=False)

# %%
rng = np.random.Generator(np.random.PCG64(seed=42))

r = np.array([0.03, 0.03, 0.006])
K = np.array([5000, 5000, 1000000])
σ = np.array([
  [0, 0, 0],
  [0, 0, 0.02],
  [0, -0.1, 0]
])
np.fill_diagonal(σ, 0)
η = np.array([0.02, 0.02, -0.1])
start = 5734 - site_data.index[0]
time_ideal = 365
time_test = 365 * 2
time = time_ideal + time_test
D_data = site_data[MOISTURE_COL][start:start + time_test + 1]
D_test = fourier_fitting(
  (D_data - D_data.min()) / (D_data.max() - D_data.min())
)
D = lambda t: 1 if t <= time_ideal else D_test(t - time_ideal)
N0 = np.array([2500, 2500, 601])

def dN_dt(t: float, N: np.ndarray):
  return (N > 0) * (
    r * N *
    (1 + (- N / K + (N @ σ) / K) / (η * D(t) + np.minimum(1, 1 - η)))
  ) + (N <= 0) * (-D(t) * N)

sol = integrate.solve_ivp(
  dN_dt, [0, time], N0,
  t_eval=np.linspace(0, time, time + 1)
)

import time
fig, ax = plt.subplots()
ax.plot(sol.t, sol.y[0], label=f'normal')
ax.plot(sol.t, sol.y[1], label=f'damaged by insects')
#ax.plot(sol.t, sol.y[2], label=f'damaged by insects')
ax.legend()
plt.savefig(f'output/plot-{time.time()}.png', dpi=320)

# %%
