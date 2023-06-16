using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Taxi_Fare_Prediction
{
    internal class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            var trainPath = "C:\\Users\\akrc2\\Downloads\\taxi-fare-train.csv";
            var testPath = "C:\\Users\\akrc2\\Downloads\\taxi-fare-test.csv";
            var modelPath = "C:\\Users\\akrc\\Downloads\\TaxiFarePrediction.zip";

            IDataView trainView = context.Data.LoadFromTextFile<InputModel>(trainPath, hasHeader: true, separatorChar: ',');

            var preprocessPipeline = context.Transforms
                .SelectColumns(nameof(InputModel.VendorId), nameof(InputModel.RateCode), nameof(InputModel.PassengerCount), nameof(InputModel.TripTime), nameof(InputModel.TripDistance), nameof(InputModel.PaymentType), nameof(InputModel.FareAmount))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_VendorId", nameof(InputModel.VendorId)))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_RateCode", nameof(InputModel.RateCode)))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_PaymentType", nameof(InputModel.PaymentType)))
                .Append(context.Transforms.CopyColumns("Label", nameof(InputModel.FareAmount)))
                .Append(context.Transforms.Concatenate("Features", "Encoded_VendorId", "Encoded_RateCode", nameof(InputModel.PassengerCount), nameof(InputModel.TripTime), nameof(InputModel.TripDistance), "Encoded_PaymentType"));

            var trainPipeline = preprocessPipeline
                .Append(context.Regression.Trainers.FastTree());

            var model = trainPipeline.Fit(trainView);

            IDataView testView = context.Data.LoadFromTextFile<InputModel>(testPath, hasHeader: true, separatorChar: ',');

            var predictions = model.Transform(testView);
            var metrics = context.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine($"RSquared Score: {metrics.RSquared:0.00}");

            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            var input = new InputModel()
            {

            };

            var prediction = predictionEngine.Predict(input);
            Console.WriteLine($"Predicted Fare: {prediction.Prediction}");

        }
    }
}
