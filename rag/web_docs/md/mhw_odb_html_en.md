**Marine Heatwaves**

[HTML source from ODB Marine Heatwaves](https://eco.odb.ntu.edu.tw/pub/MHW/?lang=en)

ODB provides the open data of marine heatwave levels, as well as monthly mean sea surface temperature (SST), SST anomalies, and thermal displacements at a 0.25° x 0.25° global resolution.

## About

Global warming and the ensuing climate crisis have led to an ongoing rise in Earth's temperatures. Unusually high temperatures during heatwaves can be deadly on land, and the oceans are equally vulnerable to this threat. When seawater remains abnormally warm for an extended period compared to historical average temperatures, it's referred to as 'marine heatwaves' (MHWs) ([Pearce et al., 2011](https://fish.gov.au/Archived-Reports/Documents/Pearce_et_al_2011.pdf); [Hobday et al., 2016](https://www.sciencedirect.com/science/article/pii/S0079661116000057)). The MHWs can cover vast areas, spanning thousands of square kilometers and lasting for weeks or even months ([Di Lorenzo and Mantua, 2016](https://www.nature.com/articles/nclimate3082)). Such prolonged warming events significantly affect marine life, ecosystems, and fisheries. To raise general awareness about MHWs, ODB provides MHWs' open data that is compiled from [NOAA's OISST](https://www.ncei.noaa.gov/products/optimum-interpolation-sst) data at a 0.25° x 0.25° resolution ([Huang et al. 2021](https://journals.ametsoc.org/view/journals/clim/34/8/JCLI-D-20-0166.1.xml)). Following [Jacox et al.'s (2020)](https://www.nature.com/articles/s41586-020-2534-z) approach, ODB evaluates MHW occurrences for each month from 1982 to the present, covering approximately 40 years (for more details, refer to the 'Method' section). The severity of MHWs is classified into distinct levels according to the criteria defined by [Hobday et al.'s (2018)](https://tos.org/oceanography/article/categorizing-and-naming-marine-heatwaves). These categorized MHW levels, as well as the monthly mean SST, SST anomalies, and thermal displacements\* caused by MHWs, collectively constitute our comprehensive global database. The datasets will receive monthly updates and are available for download. Furthermore, we provide access to [WMS](https://ecodata.odb.ntu.edu.tw/geoserver/marineheatwave/wms?service=WMS&request=GetCapabilities) and [WMTS](https://ecodata.odb.ntu.edu.tw/geoserver/gwc/service/wmts/?service=WMTS&request=getCapabilities) layers for MHWs (see hyperlinks), and the [instruction](https://oceandatabank.github.io/MHW_QGIS/) for application on QGIS.

According to the [2022 Intergovernmental Panel on Climate Change (IPCC) report](https://doi.org/10.1017/9781009325844), the frequency, intensity, and duration of global MHWs have increased and will continue to rise into the future. Given the increasingly frequency and occurrence of MHWs ([Frölicher et al. 2018](https://www.nature.com/articles/s41586-018-0383-9); [Oliver et al., 2018](https://www.nature.com/articles/s41467-018-03732-9)), ODB aims to support research on the spatiotemporal variations of MHWs, their potential impact on marine life and provide a reference for predicting and responding to MHWs.

\* Thermal Displacement: In the context of MHW, 'thermal displacement' refers to the minimum distance required to follow the long-term average sea surface temperature ([Jacox et al., 2020](https://www.nature.com/articles/s41586-020-2534-z)). Specifically, thermal displacement signifies the range over which marine heatwaves influence surrounding marine ecosystems, particularly concerning the movement, distribution range, and ecosystem function of marine organisms.

![Marine heatwaves level](https://eco.odb.ntu.edu.tw/pub/MHW/assets/hobday_2018_2.jpg)

Marine heatwaves are classified according to [Hobday et al., 2018](https://tos.org/oceanography/article/categorizing-and-naming-marine-heatwaves). Monthly SST anomalies fall into four levels: below twice the threshold, two to three times the threshold, three to four times the threshold, and above four times the threshold. These levels are labeled as Moderate, Strong, Severe, and Extreme, respectively. Figure adapted from [Hobday et al. (2018)](https://tos.org/oceanography/article/categorizing-and-naming-marine-heatwaves).

![Marine heatwaves time-series](https://eco.odb.ntu.edu.tw/pub/MHW/assets/time_series_example.jpg)

In the waters near Taiwan at longitude 122.625°E and latitude 25.375°N, the time series of the marine heatwave level from January 2020 to now.

## Method

### ODB Marine Heatwave Definition and Calculation Method

### Step A

Using NOAA OISST v2.1 dataset, calculate the monthly mean SST from 1982 to the present.

### Step B

Calculate monthly SST anomalies\* :  
Monthly mean SST (from Step A) - Long-term monthly mean SST (from 1982-2011).  
\*The result is without detrend.

### Step C

Based on the SST anomalies, define the marine heatwave threshold values by Step C-1.

### Step C-1

The threshold for marine heatwave occurrence: The monthly SST anomalies (as in Step B) must exceed the 90th percentile of the seasonal background value\*.  
\*Seasonal background value: Incorporate the month prior and subsequent to the calculated month as the background. For example, to set the threshold for marine heatwaves in February, consider the combined 90th percentile of SST anomalies for January, February, and March during 1982-2011 ([Jacox et al., 2020](https://www.nature.com/articles/s41586-020-2534-z)).

### Step D

By Step B and Step C, get the SST anomalies and determine the occurrence of marine heatwaves. If a marine heatwave occurred, evaluate the thermal displacement caused by the marine heatwave\*.  
\*Thermal displacement is defined as the minimum distance needed to track the long-term average sea surface temperature, as described by Jacox et al. (2020). In its calculation, factors like the feasibility of biological movement are taken into account, preventing unrealistic pathways such as land crossings. The global ocean is segmented into regions with certain areas designated as restricted for movement ([Jacox et al., 2020](https://www.nature.com/articles/s41586-020-2534-z)). For a comprehensive breakdown of the calculation methods, please see the provided Github link.

### Results

![2025/07 Marine Heatwave levels (Monthly 25km SST Anomaly)](https://eco.odb.ntu.edu.tw/pub/MHW/assets/202507_level.jpg)  
  
2025/07 Marine Heatwave levels (Monthly 25km SST Anomaly)
  
![2025/07 Thermal Displacement caused by Marine Heatwaves](https://eco.odb.ntu.edu.tw/pub/MHW/assets/202507_td.jpg)

2025/07 Thermal Displacement caused by Marine Heatwaves

### Reference List

Di Lorenzo, E., &Mantua, N. (2016). Multi-year persistence of the 2014/15 North Pacific marine heatwave. Nature Climate Change , 6(11), 1042–1047. [doi: 10.1038/nclimate3082](http://dx.doi.org/10.1038/nclimate3082)

Frölicher, T. L., Fischer, E. M., &Gruber, N. (2018). Marine heatwaves under global warming. Nature, 560(7718), 360–364. [doi: 10.1038/s41586-018-0383-9](https://doi.org/10.1038/s41586-018-0383-9)

Hobday, A. J., Alexander, L.V., Perkins, S. E., Smale, D. A., Straub, S. C., Oliver, E. C. J., Benthuysen, J. A., Burrows, M. T., Donat, M. G., Feng, M., Holbrook, N. J., Moore, P. J., Scannell, H. A., SenGupta, A., &Wernberg, T. (2016). A hierarchical approach to defining marine heatwaves. Progress in Oceanography, 141, 227–238. [doi: 10.1016/j.pocean.2015.12.014](https://doi.org/10.1016/j.pocean.2015.12.014)

Hobday, A. J., Oliver, E. C. J., Gupta, A.Sen, Benthuysen, J. A., Burrows, M. T., Donat, M. G., Holbrook, N. J., Moore, P. J., Thomsen, M. S., Wernberg, T., &Smale, D. A. (2018). Categorizing and naming marine heatwaves. Oceanography, 31(2), 162–173. [doi: 10.5670/oceanog.2018.205](https://doi.org/10.5670/oceanog.2018.205)

Huang, B., C. Liu, V. Banzon, E. Freeman, G. Graham, B. Hankins, T. Smith, and H.-M. Zhang, 2020: Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1, Journal of Climate, 34, 2923-2939. [doi: 10.1175/JCLI-D-20-0166.1](https://doi.org/10.1175/JCLI-D-20-0166.1)

IPCC, 2022: Climate Change 2022: Impacts, Adaptation and Vulnerability. Contribution of Working Group II to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change \[H.-O. Pörtner, D.C. Roberts, M. Tignor, E.S. Poloczanska, K. Mintenbeck, A. Alegría, M. Craig, S. Langsdorf, S. Löschke, V. Möller, A. Okem, B. Rama (eds.)\]. Cambridge University Press. Cambridge University Press, Cambridge, UK and New York, NY, USA, 3056 pp. [doi:10.1017/9781009325844](https://doi.org/10.1017/9781009325844)

Jacox, M., Alexander, M., Bograd, S., &Scott, J. (2020). Thermal displacement by marine heatwaves. Nature., 584, 82-86. [doi: 10.1038/s41586-020-2534-z](https://doi.org/10.1038/s41586-020-2534-z)

Oliver, E. C. J., Donat, M. G., Burrows, M. T., Moore, P. J., Smale, D. A., Alexander, L.V., Benthuysen, J. A., Feng, M., SenGupta, A., Hobday, A. J., Holbrook, N. J., Perkins-Kirkpatrick, S. E., Scannell, H. A., Straub, S. C., &Wernberg, T. (2018). Longer and more frequent marine heatwaves over the past century. Nature Communications, 9(1), 1–12. [doi: 10.1038/s41467-018-03732-9](https://doi.org/10.1038/s41467-018-03732-9)

Pearce, A., Lenanton, R., Jackson, G., Moore, J., Feng, M., &Gaughan, D. (2011). The “marine heat wave” off Western Australia during the summer of 2010/11. Fisheries Research Report No. 222. Department of Fisheries, Western Australia. 40pp.

## Download

Important Notes

-   The downloaded file will be in .csv format.
-   Input data columns:
    -   Longitude range: -180° ~ 180°
    -   Latitude range: -90° ~ 90°
    -   Time range: 1982-Jan-01 to the previous month
-   Output data columns:
    -   lon (longitude)
    -   lat (latitude)
    -   date (date format: yyyy-mm-dd, e.g. 2023-06-01 equivalent to data for June 2023)
    -   sst (monthly mean sea surface temperature, in °C, blank indicates NaN)
    -   sst\_anomaly (monthly sea surface temperature anomaly, in °C, blank indicates NaN)
    -   level (monthly marine heatwave level, -1 indicates sea ice for the month, 0-4 represents the degree of marine heatwave level )
    -   td (monthly thermal displacement, in km, blank indicates NaN)
-   For downloading single-point data with no time constraints, please input the starting longitude, starting latitude, and time period.
-   For downloading regional data, if the region's dimensions are ≤ 10° x 10°, the maximum time range is ten years. Please input the longitude range, latitude range, and time period.
-   For downloading regional data, if the region's dimensions are > 10° x 10°, the maximum time range is one year. Please input the longitude range, latitude range, and time period.
-   For downloading regional data, if the region's dimensions are > 90° x 90°, the maximum time range is one month. Please input the longitude range, latitude range, and time period.
-   If you utilize data provided by ODB, please cite as: Ocean Data Bank, National Science and Technology Council, Taiwan. https://doi.org/10.5281/zenodo.7512112. Accessed DAY/MONTH/YEAR from www.odb.ntu.edu.tw.

download

### Download by entering the coordinates below

Start Longitude

End Longitude

Start Latitude 

End Latitude 

Data Download (multiple)

Monthly mean SST（每月海表平均溫度） Monthly SST Anomalies（每月海表溫度距平值） Monthly MHW levels（每月海洋熱浪級數） Themal Displacements（每月熱位移）

#Connect to your software from our API  
#lon0 and lat0 are must  
#One-point MHWs without time-span limitation (data from 1982 to latest)  
#Bounding-box <= 10 x 10 in degrees: 10-years time-span limitation: e.g. /api/mhw/csv?lon0=135&lon1=140&lat0=15&lat1=30&start=2013-01-01 (data from 2013/01/01 to 2022/12/01)  
#Bounding-box > 10 x 10 in degrees: 1-year time-span limitation: e.g. /api/mhw/csv?lon0=135&lon1=150&lat0=15&lat1=30&start=2013-01-01 (data from 2013/01/01 to 2013/12/01)  
#Bounding-box > 90 x 90 in degrees: 1-month time-span limitation: e.g. /api/mhw/csv?lon0=-180&lon1=180&lat0=-90&lat1=90&start=2013-01-01 (data of 2013/01/01)  

#API request example  
https://eco.odb.ntu.edu.tw/api/mhw/csv?lon0=121&lat0=25.6&lon1=121.7&lat1=25.6&start=2023-04-01&end=2023-07-01&append=sst,sst\_anomaly,level,td 

#output example

lon,lat,date,level,sst,sst\_anomaly,td

120.875,25.375,2023-04-01,0,22.426334,0.6179904937744141,

120.875,25.375,2023-05-01,0,25.035807,0.5337085723876953,

120.875,25.375,2023-06-01,1,27.591667,1.1035003662109375,141.22131

120.875,25.375,2023-07-01,1,28.8, 1.2374820709228516,57.366695

![hidy](https://eco.odb.ntu.edu.tw/pub/MHW/assets/hidy_mhw.svg)

## Hidy Viewer

### How to Query Marine Heatwave Data in Hidy Viewer

![Hidy Viewer 選擇海洋熱浪資料示意圖](https://eco.odb.ntu.edu.tw/pub/MHW/assets/hidystep1.png)

-   Hidy Viewer：[Go to Hidy Viewer](https://odbview.oc.ntu.edu.tw/hidy/)
-   In the left menu, select ODB Data -> Marine Heatwave Levels
-   In this category, Hidy Viewer provides two data layers：
    -   Marine Heatwave Levels (default)
    -   Sea Surface Temperature Anomalies
-   Select a data layer to display its spatial distribution for corresponding month.

![Hidy Viewer 時間選擇示意圖](https://eco.odb.ntu.edu.tw/pub/MHW/assets/hidystep2.png)

-   Click on the Datetime picker on the platform to open the calendar and adjust the date.
-   ODB marine heatwave data is on a monthly scale, so the selected date does not affect the result to display. Hidy Viewer will automatically map it to the corresponding month.
-   Data Update:
    -   Updated on the 18th of each month (UTC+8 TimeZone), with data available up to the **previous month**.
    -   For example:
        -   March 10: Latest data available is from **January**.
        -   March 20: Latest data available is from **February**.

![Hidy Viewer 時間序列示意圖](https://eco.odb.ntu.edu.tw/pub/MHW/assets/hidystep3.png)

-   In addition to the default spatial view, a time series view is also available.
-   Enable the **Time Series Plot** option to display the time series of sea surface temperature anomaly at the selected location.
-   Click on the map to select a location (marked with a white dot).
-   By default, the last two years of data are shown, with adjustable start and end years.

![Hidy Viewer 時間序列示意圖](https://eco.odb.ntu.edu.tw/pub/MHW/assets/hidystep4.png)

-   X-axis: Time
-   Y-axis: Sea Surface Temperature Anomalies
-   Circle color: Represents marine heatwave levels
-   Additional Features:
    -   Select a comparison area after opening the Time Series Plot menu.
    -   Enter the longitude and latitude range of the comparison area to display:
        -   The time series (blue line) which are the mean values within the comparison area (blue box).
        -   The time series (black line) at an individual point (white dot).
    -   Download data as CSV for both the blue line (area average) and the individual point.
-   Comparison Area Limitations:
    -   Spatial range: Maximum 10° × 10°
    -   Time range: Up to 10 years
