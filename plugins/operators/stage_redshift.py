from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

class StageToRedshiftOperator(BaseOperator):
    ui_color = '#358140'
    
    @apply_defaults
    def __init__(self,
                 # Define your operators params (with defaults) here
                 # Example:
                 # redshift_conn_id=your-connection-name
                 table="staging_songs",
                 conn_id="redshift",
                 aws_credentials_id="aws_credentials",
                 s3_bucket="udacity-dend",
                 s3_key="log_data",
                 json_path="s3://udacity-dend/log_json_path.json",
                 *args, **kwargs):

        super(StageToRedshiftOperator, self).__init__(*args, **kwargs)
        # Map params here
        # Example:
        # self.conn_id = conn_id
        self.conn_id = conn_id
        self.table = table
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.aws_credentials_id = aws_credentials_id
        self.json_path = json_path
        


    def execute(self, context):
        redshift = PostgresHook(postgres_conn_id=self.redshift_conn_id)
        aws_hook = AwsHook(self.aws_credentials_id)
        credentials = aws_hook.get_credentials()

        if self.truncate:
            self.log.info(f'Truncate Redshift table {self.table}')
            redshift.run(f'TRUNCATE {self.table}')

        # SQL query parameters
        rendered_s3_key = self.s3_key.format(**context)
        s3_path = f's3://{self.s3_bucket}/{rendered_s3_key}'

        formatted_sql = StageToRedshiftOperator.copy_sql.format(
            self.table, s3_path, credentials.access_key,
            credentials.secret_key, self.data_format, self.region,
        )

        # Run query
        self.log.info('Copy data from {s3_path} to Redshift table {self.table}')
        redshift.run(formatted_sql)



