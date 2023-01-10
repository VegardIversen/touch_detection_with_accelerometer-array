# Touch detection utilizing flexural waves on a table

## Abstract
This project serves as a pre-study in the process of utilising an optical MEMS accelerometer array to detect touch localisation on a plate. For this project, the focus will be to detect touch position on a table surface by analysing the response from a vibrating actuator. Key concepts in this analysis will be tests performed on the table plate regarding its transfer function and wave carrying characteristis, along with information obtained from edge reflections an reverberation. The final product is aimed to be a cheaper alternative to especially larger touch screens, or to be mounted on plates such as tables.

## Code structure
The code is meant to be run from main.py. Which results to produce is set in generate_results.py, which has a hierarchy of results_\<setup>() --> \<result description>() --> \<more spesific result description>() --> \<functions to do data processing and visualizations>().

The functions to do data processing and visualizations are split into directories of data_processing and data_visualization, respectively.

## File paths
The path for the data to be used is set in the start of each \<result description>() function; the data is imported into a Pandas DataFrame using the csv_to_df() function, where the more general paths for the files are set.

The path for the saving of figures is set in global_constans.py.
