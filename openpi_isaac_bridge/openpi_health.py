"""Probe OpenPI's official HTTP health route without an empty WebSocket connection."""
import argparse
import http.client
import json
import math
import time


def check_openpi_health(host='127.0.0.1',port=8000,timeout=3):
    if not isinstance(host,str) or not host.strip():raise ValueError('host must be nonempty')
    if type(port) is not int or not 1<=port<=65535:raise ValueError('invalid server port')
    if not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or not 0<timeout<=10:raise ValueError('timeout must be in (0,10] seconds')
    start=time.perf_counter();connection=http.client.HTTPConnection(host,port,timeout=timeout)
    try:
        connection.request('GET','/healthz',headers={'Connection':'close'})
        response=connection.getresponse();body=response.read(4097)
        if response.status!=200 or body.strip()!=b'OK':
            raise RuntimeError(f'OpenPI /healthz expected HTTP 200 OK; received {response.status}, body={body[:160]!r}')
        return {'status':'healthy','path':'/healthz','http_status':200,'body':'OK',
            'elapsed_s':time.perf_counter()-start,'scope':'HTTP health route only; not checkpoint identity or inference validation'}
    except (OSError,http.client.HTTPException) as error:
        raise RuntimeError(f'OpenPI health request failed at {host}:{port}: {error}') from error
    finally:connection.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8000)
    args=parser.parse_args()
    try:print(json.dumps(check_openpi_health(args.host,args.port),indent=2));return 0
    except (ValueError,RuntimeError) as error:print(str(error));return 1


if __name__=='__main__':raise SystemExit(main())
