cscript

local 0 ",`0'"
di "`0'"

syntax [anything] , [state(string asis) date(string asis) show]

if "`date'" == "" {
    local td = date(c(current_date), "DMY") - 1
	local date : di %tdNN-DD-CCYY `td'	
}

di "`state'"
di "`date'"

python:
import numpy as np
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/"\
					"COVID-19/master/csse_covid_19_data/"\
					"csse_covid_19_daily_reports/`date'.csv",
					dtype={"fips" : np.int32})
df.columns = df.columns.str.lower()
df = df.loc[df['country_region'] == "US"]
end

if "`state'" != "" {
python : df = df.loc[df['province_state'] == "`state'"]     
}


python:
df.count()
end


quietly python:
from urllib.request import urlopen
import numpy as np
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


import plotly.express as px
fig = px.choropleth(df, geojson=counties, locations='fips', color='confirmed',
						   hover_data=['combined_key', 'confirmed'],
						   color_continuous_scale='ylorrd',
						   range_color = [100, 5000],
						   scope="usa",
                           labels={'confirmed':'confirmed cases'}
                          )

from sfi import Macro
if Macro.getLocal('state'): 						  
	fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

if Macro.getLocal('show'): 						  
	fig.show()
else: 
	fig.write_html("stata/`date'-`state'.html")
end
