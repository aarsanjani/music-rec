import boto3
import argparse
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])
import sagemaker
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.predictor import RealTimePredictor
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker import get_execution_role
from sagemaker.model_monitor import CronExpressionGenerator
from time import gmtime, strftime


# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument('--baseline-data-uri', type=str)
parser.add_argument('--bucket-name', type=str)
parser.add_argument('--bucket-prefix', type=str)
parser.add_argument('--region', type=str)
parser.add_argument('--endpoint', type=str)
args = parser.parse_args()

region = args.region
bucket = args.bucket_name
prefix = args.bucket_prefix
endpoint_name = args.endpoint

boto_session = session.Session(boto3.Session())
sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)
sagemaker_session = sagemaker.session.Session(
    boto_session=boto3.Session(),
    sagemaker_client=sagemaker_client
)
role = get_execution_role(sagemaker_session=sagemaker_session)


# Enable real-time inference data capture

s3_capture_upload_path = f's3://{bucket}/{prefix}/endpoint-data-capture/' #example: s3://bucket-name/path/to/endpoint-data-capture/

# Change parameters as you would like - adjust sampling percentage, 
#  chose to capture request or response or both
data_capture_config = DataCaptureConfig(
    enable_capture = True,
    sampling_percentage=50,
    destination_s3_uri=s3_capture_upload_path,
    kms_key_id=None,
    capture_options=["REQUEST", "RESPONSE"],
    csv_content_types=["text/csv"],
    json_content_types=["application/json"]
)

# only setup model monitor once model endpoint has been created
while not sagemaker_boto_client.list_endpoints(NameContains=endpoint_name)['Endpoints']:
    time.sleep(60)
    print("Waiting on model endpoint {}".format(endpoint_name))

    
# Now it is time to apply the new configuration and wait for it to be applied
predictor = RealTimePredictor(endpoint_name=endpoint_name)
predictor.update_data_capture_config(data_capture_config=data_capture_config)
sagemaker_session.wait_for_endpoint(endpoint=endpoint_name)

print('Created RealTimePredictor at endpoint {}'.format(endpoint_name))

baseline_data_uri = args.baseline_data_uri ##'s3://bucketname/path/to/baseline/data' - Where your validation data is
baseline_results_uri = f's3://{bucket}/{prefix}/baseline/results' ##'s3://bucketname/path/to/baseline/data' - Where the results are to be stored in

print('Baseline data is at {}'.format(baseline_data_uri))

my_default_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

my_default_monitor.suggest_baseline(
    baseline_dataset=baseline_data_uri,
    dataset_format=DatasetFormat.csv(header=False),
    output_s3_uri=baseline_results_uri,
    wait=True
)

print('Model data baseline suggested at {}'.format(baseline_results_uri))

# Setup daily Cron job schedule 
mon_schedule_name = 'music-recommender-daily-monitor'
s3_report_path = f's3://{bucket}/{prefix}/monitor/report'

try:
    my_default_monitor.create_monitoring_schedule(
        monitor_schedule_name=mon_schedule_name,
        endpoint_input=endpoint_name,
        output_s3_uri=s3_report_path,
        statistics=my_default_monitor.baseline_statistics(),
        constraints=my_default_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=True,
    )
except ValueError as e:
    print(e)

desc_schedule_result = my_default_monitor.describe_schedule()
print('Created monitoring schedule. Schedule status: {}'.format(desc_schedule_result['MonitoringScheduleStatus']))