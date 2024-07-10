from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle

def create_app():
    app = Flask(__name__)

    # Load preprocessed pipeline
    preprocessing_pipe = pickle.load(open('pipepreprocessing.pkl', 'rb'))
    pipe_uval = pickle.load(open('pipe_u_val.pkl', 'rb'))
    # Load Keras model
    keras_model = load_model('my_keras_model.keras')

    @app.route("/")
    def hello_world():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict_savings():
        error_message = None
        try:
            climaticZone = request.form.get('ClimaticZone')
            shadeDirection = request.form.get('ShadeDirection')
            buildingOrientation = int(request.form.get('BuildingOrientation'))
            latitude = float(request.form.get('Latitude'))
            longitude = float(request.form.get('Longitude'))
            shadeExtent = float(request.form.get('ShadeExtent'))
            roofUValue = float(request.form.get('RoofUValue'))
            heightOfShade = int(request.form.get('HeightofShade'))
            shadingTransmittance = float(request.form.get('ShadingTransmittance'))

            if not (0 < heightOfShade <= 8):
                error_message = 'Error: Height Of Shade value must be between 1 and 8. Please write values from 0 to 8.'
            elif not (0 <= shadingTransmittance <= 1):
                error_message = 'Error: Shading Transmittance value must be between 0 and 1. Please write values from 0 to 1.'
            elif not (0 <= shadeExtent <= 1):
                error_message = 'Error: Shade extent value must be between 0 and 1. Please write values from 0 to 1.'

            if error_message:
                raise ValueError(error_message)

            def combined_nearest_values(climaticZone, value1, value2):
                data = {
                    'City': ['Trivandrum', 'Tiruchirapalli', 'Mangalore', 'Bengaluru', 'Chennai', 'Bellary', 'Panjim', 'Hyderabad', 'Solapur', 'Visakhapatnam', 'Pune', 'Mumbai', 'Aurangabad', 'Nagpur', 'Surat', 'Kolkata', 'Ahmedabad', 'Bhopal', 'Udaipur', 'Shillong', 'Lucknow', 'Jaipur', 'Bikaner', 'New Delhi', 'Dehradun', 'Amritsar'],
                    'Climatic Zone': ['Warm and Humid', 'Warm and Humid', 'Warm and Humid', 'Temperate', 'Warm and Humid', 'Composite', 'Warm and Humid', 'Composite', 'Hot and Dry', 'Warm and Humid', 'Warm and Humid', 'Warm and Humid', 'Hot and Dry', 'Composite', 'Hot and Dry', 'Warm and Humid', 'Hot and Dry', 'Composite', 'Hot and Dry', 'Warm and Humid', 'Composite', 'Hot and Dry', 'Hot and Dry', 'Composite', 'Composite', 'Composite'],
                    'Latitude': [8.51, 10.79, 12.92, 12.97, 13.08, 15.14, 15.49, 17.38, 17.67, 17.69, 18.52, 19.07, 19.88, 21.15, 21.17, 22.57, 23.03, 23.26, 24.58, 25.57, 26.85, 26.92, 28.02, 28.61, 30.32, 31.64],
                    'Longitude': [76.9366, 78.7047, 74.856, 77.5946, 80.2707, 76.9214, 73.8278, 78.4867, 75.9064, 83.2185, 73.8567, 72.8777, 75.3433, 79.0882, 72.8311, 88.3639, 72.5714, 77.4126, 73.7125, 91.8933, 80.9462, 75.7873, 73.3119, 77.209, 78.0322, 74.8737]
                }
                df = pd.DataFrame(data)
                same_zone_cities = df[df['Climatic Zone'] == climaticZone]

                def euclidean_distance(lat1, lon1, lat2, lon2):
                    return ((lat1 - lat2)**2 + (lon1 - lon2)**2) ** 0.5

                same_zone_cities['Distance'] = same_zone_cities.apply(lambda row: euclidean_distance(value1, value2, row['Latitude'], row['Longitude']), axis=1)

                # Find the nearest city
                nearest_city_info = same_zone_cities.sort_values(by='Distance').iloc[0]

                return nearest_city_info['Latitude'], nearest_city_info['Longitude'], nearest_city_info['City']

            latitude, longitude, city = combined_nearest_values(climaticZone, latitude, longitude)

            # Preprocess input data
            input_data = np.array([climaticZone, shadeDirection, buildingOrientation, latitude, longitude, shadeExtent, roofUValue, heightOfShade, shadingTransmittance]).reshape(1, -1)
            preprocessed_input = preprocessing_pipe.transform(input_data)
            preprocessed_input = preprocessed_input.astype(np.float64)

            # Make predictions
            result = keras_model.predict(preprocessed_input)
            input_data_for_uval = np.array([climaticZone, buildingOrientation, latitude, longitude, result[0][0]]).reshape(1, -1)
            result_uval = pipe_uval.predict(input_data_for_uval)

            return render_template(
                'predict.html',
                latitude=latitude,
                longitude=longitude,
                climaticZone=climaticZone,
                shadeDirection=shadeDirection,
                buildingOrientation=buildingOrientation,
                shadeExtent=shadeExtent,
                roofUValue=roofUValue,
                heightOfShade=heightOfShade,
                shadingTransmittance=shadingTransmittance,
                result=round(result[0][0], 2),
                result_uval=round(result_uval[0], 2),
                error_message=None,
                city=city,
            )
        except (TypeError, ValueError) as e:
            return render_template(
                'index.html',
                climaticZone=request.form.get('ClimaticZone'),
                shadeDirection=request.form.get('ShadeDirection'),
                buildingOrientation=request.form.get('BuildingOrientation'),
                latitude=request.form.get('Latitude'),
                longitude=request.form.get('Longitude'),
                shadeExtent=request.form.get('ShadeExtent'),
                roofUValue=request.form.get('RoofUValue'),
                heightOfShade=request.form.get('HeightofShade'),
                shadingTransmittance=request.form.get('ShadingTransmittance'),
                error_message=str(e),
                city=None,
            )

    return app

if __name__ == '__main__':
    from waitress import serve

    app = create_app()
    serve(app, host="127.0.0.1", port=8080)
