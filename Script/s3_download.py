from boto3.session import Session

def s3_download():
    ACCESS_KEY_ID='AKIAYQADVMBG4S6NS65O'
    SECRET_ACCESS_KEY='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'


    session=Session(aws_access_key_id=ACCESS_KEY_ID,aws_secret_access_key=SECRET_ACCESS_KEY)
    print(session)

    s3=session.resource('s3')

    bucket='projectbank1'

    my_bucket=s3.Bucket(bucket)
    print(my_bucket)

    #Check Which File Availabe in our S3 Bucket


    print('Files Available In S3 Buckets')
    for s3_files in my_bucket.objects.all():
        print(s3_files.key)  


    # File Download
    #my_bucket.download_file(' file name','File Path)


    my_bucket.download_file('bank-full.csv',r"D:\Data science\complete projects\project_bank\Data\Raw_data\bank-full.csv")


    print('file downloded '.center(50, '*'))

s3_down = s3_download()
