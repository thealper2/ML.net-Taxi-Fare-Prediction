using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Taxi_Fare_Prediction
{
    internal class ResultModel
    {
        [ColumnName("Score")]
        public float Prediction;
    }
}
