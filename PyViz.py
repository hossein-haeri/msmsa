import pandas as pd
import holoviews as hv
from holoviews.operation.datashader import datashade
import datashader as ds
# from bokeh.io import output_notebook

# Enable the HoloViews extension with Bokeh backend
hv.extension('bokeh')
# output_notebook()

# Load your dataset
df = pd.read_csv('datasets/Teconer_2018_Jan_light_100K.csv')
# df = pd.read_csv('datasets/Teconer_Downtown_1M.csv')

# Create a HoloViews Points object
# Replace 'Latitude' and 'Longitude' with the actual column names in your CSV file
points = hv.Points(df, ['Latitude', 'Longitude'])

# Use Datashader to handle large data efficiently
shaded = datashade(points)

# Customize the plot
shaded.opts(width=800, height=600, tools=['hover'])

# Save the plot as an HTML file
hv.save(shaded, 'interactive_plot.html', backend='bokeh')

# save as an html file


# Display the interactive plot
shaded
