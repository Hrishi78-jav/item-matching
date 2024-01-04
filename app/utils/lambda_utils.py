import json
import boto3
import botocore.config


def invoke_lambda(lambda_name, input_data, aws_region='ap-south-1', invocation_type='sync', max_retries=None, timeout=None):
    """

    :param str lambda_name: name of lambda
    :param dict input_data: input event
    :param str aws_region:
    :param str invocation_type: sync if output is required, async won't wait for lambda to run
    :param int max_retries:
    :param int timeout:
    :return: dict output of lambda invocation
    """
    input_data = json.dumps(input_data)
    print(f'Invoking {lambda_name} lambda')
    try:
        config = botocore.config.Config(
            read_timeout=timeout if timeout is not None else 900,
            retries={'max_attempts': max_retries if max_retries is not None else 0}
        )
        lambda_client = boto3.client('lambda', region_name=aws_region, config=config)

        invocation_type = 'Event' if invocation_type == 'async' else 'RequestResponse'
        response = lambda_client.invoke(FunctionName=lambda_name,
                                        InvocationType=invocation_type,
                                        Payload=input_data)
        if response['StatusCode'] in range(200, 300):
            response = response['Payload'].read()
            if invocation_type != 'Event':
                response = json.loads(response)
            else:
                response = ""
    except Exception as e:
        raise Exception(f'{lambda_name} lambda Invocation Failed: ' + str(e))

    return response
