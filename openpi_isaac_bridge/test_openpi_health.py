import http.server
import threading
import unittest
from unittest.mock import MagicMock,patch
from openpi_health import check_openpi_health


class HealthTests(unittest.TestCase):
    def test_sends_a_real_http_health_request(self):
        seen=[]
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append((self.command,self.path,self.headers.get('Connection')))
                self.send_response(200);self.end_headers();self.wfile.write(b'OK\n')
            def log_message(self,*args):pass
        with http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:r=check_openpi_health('127.0.0.1',server.server_port)
            finally:server.shutdown();thread.join()
        self.assertEqual(r['status'],'healthy');self.assertEqual(seen,[('GET','/healthz','close')])

    def test_wrong_endpoint_or_body_does_not_confirm_openpi(self):
        for status,body in [(404,b'not found'),(426,b'upgrade required'),(200,b'<html>other server</html>')]:
            with self.subTest(status=status):
                client=MagicMock();response=client.getresponse.return_value;response.status=status;response.read.return_value=body
                with patch('openpi_health.http.client.HTTPConnection',return_value=client),self.assertRaises(RuntimeError):check_openpi_health()
                client.close.assert_called_once()

    def test_network_failure_fails_before_policy_client_retry(self):
        client=MagicMock();client.request.side_effect=ConnectionRefusedError('offline')
        with patch('openpi_health.http.client.HTTPConnection',return_value=client),self.assertRaisesRegex(RuntimeError,'health request failed'):check_openpi_health()
        client.close.assert_called_once()

    def test_invalid_connection_settings_rejected(self):
        for args in [('',8000,3),('localhost',0,3),('localhost',True,3),('localhost',8000,float('nan'))]:
            with self.subTest(args=args),self.assertRaises(ValueError):check_openpi_health(*args)


if __name__=='__main__':unittest.main()
