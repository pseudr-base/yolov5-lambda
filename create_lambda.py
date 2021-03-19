import boto3, datetime, subprocess, json

now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d%H%M%S')
image = 'lambda-container-yolov5'
tag = ':latest'
repository_name = f'{image}-{now}'

ecr = boto3.client('ecr')
response = ecr.create_repository(
    repositoryName=repository_name,
    imageScanningConfiguration={'scanOnPush': True},
)

uri = response['repository']['repositoryUri']
account_id = response['repository']['registryId']
region = uri.split('.')[3]

# build
cmd = f'docker build -t {image} .'
print(f'exec: {cmd}')
subprocess.Popen(cmd.split(" "))

# tag
cmd = f'docker tag {image}{tag} {uri}{tag}'
print(f'exec: {cmd}')
subprocess.Popen(cmd.split(' '))

# login
domain = uri.split('/')[0]
cmd1 = "aws ecr get-login-password"
cmd2 = f"docker login --username AWS --password-stdin {domain}"
res = subprocess.Popen(cmd1.split(' '), stdout=subprocess.PIPE)
subprocess.check_output(cmd2.split(' '), stdin=res.stdout)


# push
cmd = f'docker push {uri}{tag}'
print(f'exec: {cmd}')
subprocess.Popen(cmd.split(' '))


iam = boto3.client('iam')
function_name = f'{image}-function'

doc = {
    'Version': '2012-10-17',
    'Statement': [
        {
            'Action': 'sts:AssumeRole',
            'Principal': {
                'Service': 'lambda.amazonaws.com'
                
            },
            'Effect': 'Allow',
            'Sid': ''
            
        }
    ]
}
role_name = f'{image}-role'
res = iam.create_role(
    Path = '/service-role/',
    RoleName=role_name,
    AssumeRolePolicyDocument=json.dumps(doc),
    Description=f'exec role',
    MaxSessionDuration=3600*12
)

role_arn = res['Role']['Arn']

doc = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "logs:CreateLogGroup",
            "Resource": f"arn:aws:logs:{account_id}:*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": [
                f"arn:aws:logs:{region}:{account_id}:log-group:/aws/lambda/{function_name}:*"
            ]
        }
    ]
}

poicy_name = f'{image}-policy'
res = iam.create_policy(
    PolicyName=poicy_name,
    PolicyDocument=json.dumps(doc),
)
policy_arn = res['Policy']['Arn']

res = iam.attach_role_policy(
    RoleName=role_name,
    PolicyArn=policy_arn
)

lambda_client = boto3.client('lambda')
ecr = boto3.client('ecr')
res = ecr.describe_images(
    repositoryName = repository_name
)
image_digest = res['imageDetails'][0]['imageDigest']

lambda_client.create_function(
    FunctionName=function_name,
    Role=role_arn,
    Code={
        'ImageUri': f'{uri}@{image_digest}'
    },
    Description='input-> b64img, output -> b64img, yolov5 detect',
    Timeout=10,
    MemorySize=1024,
    Publish=True,
    PackageType='Image',
)
