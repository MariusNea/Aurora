# plugins/statistics_plugin.py
# this plugin can be accessed from Statistics menu inside AURORA's GUI
# The MyApp class now supports creating two separate menu categories: "Statistics" and "Machine Learning".
# The register_plugin method requires a category argument to determine under which menu the plugin should be registered.
# Plugins need to specify their category ('statistics' or 'machine_learning') when using the register_plugin decorator.
# The code can use "app.get_dataframe()" method to access the main dataframe from AURORA.

def register(app):
    @app.register_plugin('category', 'stats_test', 'Perform Stats Test')
    def stats_test():
       
		# You can add your code here
        print("Running a statistics or machine learning test...")
		
# stats_test - is function's name.
# "Perform Stats Test" - is the text that will apear in AURORA's GUI

