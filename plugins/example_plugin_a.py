# plugins/example_plugin_a.py
# The MyApp class now supports creating two separate menu categories: "Statistics" and "Machine Learning".
# The register_plugin method requires a category argument to determine under which menu the plugin should be registered.
# Plugins need to specify their category ('statistics' or 'machine_learning') when using the register_plugin decorator.
# The code can use "app.get_dataframe()" method to access the main dataframe from AURORA.


#####################################################
#### Package: Aurora
#### Plugin: Test plugin
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################


def register(app):
    @app.register_plugin('category', 'stats_test', 'Perform Stats Test')
    def stats_test():
       
		# You can add your code here
        print("Running a statistics or machine learning test...")

# category - replace this with 'statistics' or 'machine_learning'	
# stats_test - is function's name.
# "Perform Stats Test" - is the text that will apear in AURORA's GUI

