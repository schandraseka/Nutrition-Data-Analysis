__author__ = "Siddharth Chandrasekaran"
__license__ = "GPL"
__version__ = "1.0.1"
__email__ = "schandraseka@umass.edu"


from bokeh.core.properties import field
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import (ColumnDataSource, HoverTool, SingleIntervalTicker, Slider, CategoricalColorMapper, Button, FactorRange, LinearColorMapper, ColorBar, LogColorMapper)
from bokeh.palettes import Category10, Category20, Plasma256, Paired, linear_palette, Viridis256,Greys256
from bokeh.plotting import figure	
from bokeh.io import output_notebook
from bokeh.charts import HeatMap, show, bins
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider
from functools import partial
from bokeh.models.widgets import DataTable
from bokeh.models.glyphs import HBar
from sklearn import linear_model
from itertools import product
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from bokeh.models import Legend
import time

data = pd.read_csv("nutrition_raw_anonymized_data.csv")
data.replace(('Innie', 'Outie'), ('No', 'Yes'), inplace=True)

diseases = data.iloc[:,1:4].replace(('No', 'Yes'), (0, 1))
#display(diseases)
characteristics = data.iloc[:,5:27].replace(('No', 'Yes'), (0, 1))
#display(habits)
foodhabits = data.iloc[:,28:1093]
#display(foodhabits)

def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

#Parse param grid
def parse_param_grid(param_grid):
    for p in param_grid:
            items = sorted(p.items())
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params
                
#Only prints out the top n results based on out of sample error
def get_nbest_result(results,n):
    results.sort(key=lambda tup: tup[0])
    return results[0]

#Grid Search CV custom
def cross_validation(basemodel, cv, X, y, paramgridIterator):
    result = []
    X = X.as_matrix()
    for param in paramgridIterator:
        start_time = time.time()
        print("Starting : "+ str(param))
        validation_folds_score = []
        kf = KFold(n_splits=cv, random_state=1, shuffle=True)
        for train_index, test_index in kf.split(X):
            model = basemodel.set_params(**param)
            
            y_train, y_test = y[train_index], y[test_index]
            X_train, X_test = X[train_index], X[test_index]
            fittedmodel = model.fit(X_train, y_train)
            ycap = fittedmodel.predict(X_test)
            validation_folds_score.append(compute_error(ycap, y_test))
        result.append((sum(validation_folds_score)/float(len(validation_folds_score)), param))
        print("--- %s seconds ---" % (time.time() - start_time))
    return result

def classification(X, algorithm, y):

    if algorithm=='LinearSVC':
        param_grid = {"penalty": ["l2"],
              "class_weight": ["balanced"],
		      "solver": ["liblinear","newton-cg" ],																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																			
		      }
        param_grid = dict(param_grid)
        paramgridIterator = parse_param_grid([param_grid])
        lr = linear_model.LogisticRegression()
        nbesttreemodel = get_nbest_result(cross_validation(lr, 3, X, y, paramgridIterator), 1)
        print((nbesttreemodel[1]))
        fittedmodel = lr.set_params(**nbesttreemodel[1]).fit(X, y)
        return list(fittedmodel.coef_[0])
    elif algorithm=='RandomForest':
        param_grid = {"criterion": ["gini", "entropy"],																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																		
		      }
        param_grid = dict(param_grid)
        paramgridIterator = parse_param_grid([param_grid])
        lr = ExtraTreesClassifier()
        nbesttreemodel = get_nbest_result(cross_validation(lr, 3, X, y, paramgridIterator), 1)
        print((nbesttreemodel[1]))
        fittedmodel = lr.set_params(**nbesttreemodel[1]).fit(X, y)
        return list(fittedmodel.feature_importances_ )
        
    elif algorithm=='DecisionTree':
        param_grid = {"criterion": ["gini", "entropy"],																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																		
		      }
        param_grid = dict(param_grid)
        paramgridIterator = parse_param_grid([param_grid])
        lr = DecisionTreeClassifier()
        nbesttreemodel = get_nbest_result(cross_validation(lr, 3, X, y, paramgridIterator), 1)
        print((nbesttreemodel[1]))
        fittedmodel = lr.set_params(**nbesttreemodel[1]).fit(X, y)
        return list(fittedmodel.feature_importances_)

train_data = pd.concat([characteristics, foodhabits], axis=1)

selected1 = "LinearSVC"
selected2 = "cancer"
test_data = diseases[selected2]

feature_importance = classification(train_data, selected1, test_data)
histogram = pd.DataFrame(
    {'Features': list(train_data.columns),
    'Importance': list(feature_importance),
    })

#Take only values > 0

histogram = histogram.loc[histogram['Importance'] > 0.01]
def update(attrname, old, new):
	selected1 = select1.value
	selected2 = select2.value
	test_data = diseases[selected2] 
	feature_importance = classification(train_data, selected1, test_data)
	histogram = pd.DataFrame(
    		{'Features': list(train_data.columns),
    		'Importance': list(feature_importance),
    		})
	histogram = histogram.loc[histogram['Importance'] > 0.01]
	print(histogram)
	source = ColumnDataSource(data=histogram)
	p = figure(x_range= list(source.data['Features']), plot_height=300, plot_width = 800, title="Feature importance per selected disease for a given classification algorithm")
	p.vbar(x='Features', top='Importance', width=0.9,source=source)
	p.xaxis.axis_label = "Features with importance > 0.01 in predicting the disease"
	p.yaxis.axis_label = "Importance of the feature (0-1) in the various classification algorithms"
	p.vbar(x='Features', top='Importance', width=0.9,source=source)
	p.add_tools(HoverTool(tooltips=[("Feature Name", "@Features"), ("Value", "@Importance")]))
	layout.children[1] = p
 
	
source = ColumnDataSource(data=histogram)
p = figure(x_range= list(source.data['Features']), plot_height=300, plot_width = 800, title="Feature importance per selected disease for a given classification algorithm")
p.xaxis.axis_label = "Features with importance > 0.01 in predicting the disease"
p.yaxis.axis_label = "Importance of the feature (0-1)"
p.vbar(x='Features', top='Importance', width=0.9,source=source)
p.add_tools(HoverTool(tooltips=[("Feature Name", "@Features"), ("Value", "@Importance")]))


select1 = Select(title="Choose the classification algorithm to be run on this dataset:", value=selected1, options=["LinearSVC", "RandomForest", "DecisionTree"])


select2 = Select(title="Choose the disease you'd like to classfication algorithm to be trained on :", value=selected2, options=["cancer", "diabetes", "heart_disease"])


	
select1.on_change('value', update)
select2.on_change('value', update)


normalized_data = pd.concat([diseases, train_data], axis=1)
correlation_matrix = normalized_data.corr()
cols = pd.Series(correlation_matrix.columns.tolist())
correlation_matrix['Feature1'] = cols.values
#display(correlation_matrix)
correlation_matrix  = correlation_matrix .set_index('Feature1')
correlation_matrix .columns.name = 'Feature2'
correlation_matrix  = pd.DataFrame(correlation_matrix .stack(), columns=['Correlation']).reset_index()

correlation_matrix = correlation_matrix.loc[correlation_matrix['Feature1'].isin(list(diseases.columns))]
correlation_matrix = correlation_matrix.loc[~(correlation_matrix['Feature2'].isin(list(diseases.columns)))]
#display(correlation_matrix.head(1500))






low = correlation_matrix['Correlation'].min()
high=correlation_matrix['Correlation'].max()
#mapper = LogColorMapper(palette=linear_palette(Greys256,50), low=abs(correlation_matrix['Correlation'].min()), high=abs(correlation_matrix['Correlation'].max()))
mapper = LogColorMapper(palette=["#000000","#4169E1","#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d", "#000000"], low=low, high=high)
source1 = ColumnDataSource(correlation_matrix)
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
#print(list(set(list(correlation_matrix['Feature_2']))))
p1 = figure(title="Heat map of correlation matrix between the diseases and the various other fields",  plot_width=900, plot_height=1800,x_axis_location="above",
           x_range=list(set(list(correlation_matrix['Feature1']))), y_range = list(set(list(correlation_matrix['Feature2']))), tools=TOOLS)
p1.rect(x="Feature1", y="Feature2", width=1, height=1,source=source1,fill_color={'field': 'Correlation', 'transform': mapper},line_color=None)

p1.grid.grid_line_color = None
p1.axis.axis_line_color = None
p1.axis.major_tick_line_color = None
p1.xaxis.major_label_text_font_size = "12pt"
p1.yaxis.major_label_text_font_size = "4pt"
p1.axis.major_label_standoff = 0
p1.xaxis.major_label_orientation = 22 / 21
color_bar = ColorBar(color_mapper=mapper)

p1.add_layout(color_bar)
p1.xaxis.axis_label = "Diseases"
p1.yaxis.axis_label = "All features"

layout = layout([
    [select1,select2],[p],[p1]
])


p1.select_one(HoverTool).tooltips = [
     ('Feature 1', '@Feature1'),
     ('Feature 2', '@Feature2'),
     ('Correlation', '@Correlation')
]
curdoc().add_root(layout)
curdoc().title = "690V Assignment - Classification"
show(layout)


