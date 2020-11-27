from setuptools import setup


setup(name='cars_price',
      version='0.1',
      description='Cars price prediction model',
      url='https://github.com/konorbj/CarsPricePrediction',
      author='Daniil Maksymenko',
      author_email='daniil.maksymenko@nure.ua',
      license='MIT',
      packages=['CarsPricePrediction'],
      install_requires=[
          'numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm'
      ],
      package_data = {'CarsPricePrediction': ['utils/*']},
      data_files=[('utils', ['./CarsPricePrediction/utils/train.csv', './CarsPricePrediction/utils/zipcodes.csv', 
      './CarsPricePrediction/utils/checksum.sha256', './CarsPricePrediction/utils/model.pkl',
      './CarsPricePrediction/utils/geo_clustering.pkl', './CarsPricePrediction/utils/label_encoder_brand.pkl',
      './CarsPricePrediction/utils/label_encoder_fuel.pkl', './CarsPricePrediction/utils/label_encoder_gearbox.pkl',
      './CarsPricePrediction/utils/label_encoder_model.pkl', './CarsPricePrediction/utils/label_encoder_type.pkl'])],
      include_package_data=True,
      zip_safe=False)
