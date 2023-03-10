from boto3.session import Session

def s3_upload():
    ACCESS_KEY_ID='AKIAYQADVMBG4S6NS65O'
    SECRET_ACCESS_KEY='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'


    session=Session(aws_access_key_id=ACCESS_KEY_ID,aws_secret_access_key=SECRET_ACCESS_KEY)


    s3=session.resource('s3')

    bucket='projectbank1'

    my_bucket=s3.Bucket(bucket)
    print(my_bucket)


    file_uploded=r"D:\Data science\complete projects\project_bank\Data\clean_df.csv"
    obj_name='clean_df.csv'


    my_bucket.upload_file(file_uploded,obj_name)
    print('File uploded')
    
    return s3_upload
 # s3_upload()