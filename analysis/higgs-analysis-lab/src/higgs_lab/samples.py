"""Open Data process groups used by preparation."""
SAMPLES = [
    {'name': 'Data', 'role': 'data', 'dids': ['data']},
    {'name': 'Reducible/other background', 'role': 'background',
     'dids': [410470, 410155, 410218, 410219, 410220, 410408, 412043,
              364242, 364243, 364245, 364246, 364247, 364248,
              700601, 700320, 700321, 700322, 700323, 700324, 700325],
     'color': '#6b59d3'},
    {'name': 'Irreducible $ZZ^{*}$', 'role': 'background',
     'dids': [700600, 700587, 700591], 'color': '#ff0000'},
    {'name': 'SM Higgs ($m_H=125$ GeV)', 'role': 'signal',
     'dids': [345060, 346228, 345066,
              346645, 346646, 346647,
              346340, 346341, 346342,
              346414, 346511],
     'color': '#00cdff'},
]
