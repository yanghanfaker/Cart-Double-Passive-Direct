import math
from pathlib import Path
import tempfile
import unittest
from camera_reference_audit import camera_from_world,rotate_pixel_180,local_mount,angle,overlay


class CameraReferenceTests(unittest.TestCase):
    def test_project_ros_and_rotate_pixel_involution(self):
        import numpy as np
        K=[[100,0,128],[0,100,128],[0,0,1]]
        uv=camera_from_world([.1,.2,1],[0,0,0],np.eye(3),K)
        self.assertEqual(uv,[138,148])
        self.assertEqual(rotate_pixel_180(uv,256,256),[117,107])
        self.assertEqual(rotate_pixel_180(rotate_pixel_180(uv,256,256),256,256),uv)
        self.assertIsNone(camera_from_world([0,0,-1],[0,0,0],np.eye(3),K))
        self.assertIsNone(rotate_pixel_180(None,256,256))

    def test_hand_local_mount_is_invariant_under_world_transform(self):
        import numpy as np
        R=np.array([[0,-1,0],[1,0,0],[0,0,1]]);p=np.array([1.,2.,3.])
        cp=p+R@np.array([.05,0,0]);cR=R.copy()
        localp,localR=local_mount(p,R,cp,cR)
        np.testing.assert_allclose(localp,[.05,0,0],atol=1e-12)
        np.testing.assert_allclose(localR,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(angle(R),90)

    def test_pixel_projection_invariant_to_rigid_world_registration(self):
        import numpy as np
        point=np.array([.2,.1,2]);cp=np.array([0.,0.,1]);R=np.eye(3);K=np.diag([100.,100.,1.])
        worldR=np.array([[0,0,1],[0,1,0],[-1,0,0]]);t=np.array([3.,-1.,.5])
        first=camera_from_world(point,cp,R,K)
        second=camera_from_world(worldR@point+t,worldR@cp+t,worldR@R,K)
        np.testing.assert_allclose(first,second,atol=1e-12)

    def test_overlay_embeds_original_png_and_records_rotated_hypothesis(self):
        import base64
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'overlay.svg';payload=b'unchanged original image'
            overlay(path,payload,{'visible':[138,148],'behind':None,'outside':[500,500]},True)
            text=path.read_text();self.assertIn(base64.b64encode(payload).decode(),text)
            self.assertIn('cx="117"',text);self.assertIn('cy="107"',text)
            self.assertEqual(text.count('<circle'),1)


if __name__=='__main__':unittest.main()
