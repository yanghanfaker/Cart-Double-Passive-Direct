"""Read-only authored collider bounds transformed with live physics body poses.

AABB overlap/proximity is not an exact mesh contact, penetration or force test.
"""
import itertools
import re
import numpy as np


def corners(minimum,maximum):
    lo=np.asarray(minimum,dtype=float);hi=np.asarray(maximum,dtype=float)
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or (hi<lo).any():raise ValueError('invalid collider bound')
    return np.array(list(itertools.product(*zip(lo,hi))))


def live_bound(local,position,quat):
    from kinematic_attribution import rotate
    points=np.asarray(local,dtype=float);p=np.asarray(position,dtype=float)
    if points.shape!=(8,3) or p.shape!=(3,) or not np.isfinite(points).all() or not np.isfinite(p).all():raise ValueError('invalid live geometry')
    rotation=np.array([rotate(quat,axis) for axis in np.eye(3)]).T
    if not np.isfinite(rotation).all():raise ValueError('invalid live body orientation')
    world=points@rotation.T+p
    return {'minimum_m':world.min(axis=0).tolist(),'maximum_m':world.max(axis=0).tolist()}


def rigid_scale(linear):
    matrix=np.asarray(linear,dtype=float)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all():raise ValueError('invalid body scale')
    scale=np.linalg.norm(matrix,axis=1)
    if min(scale)<=1e-10:raise ValueError('invalid body scale')
    unit=matrix/scale[:,None]
    if np.linalg.det(unit)<0 or not np.allclose(unit@unit.T,np.eye(3),atol=1e-5,rtol=0):raise ValueError('sheared/reflected rigid body unsupported')
    return scale


def box_relation(a,b):
    alo=np.array(a['minimum_m']);ahi=np.array(a['maximum_m']);blo=np.array(b['minimum_m']);bhi=np.array(b['maximum_m'])
    gap=np.maximum(alo-bhi,blo-ahi)
    xy=bool(np.all(gap[:2]<=0))
    return {'aabb_distance_m':float(np.linalg.norm(np.maximum(gap,0))),
        'aabb_overlap':bool(np.all(gap<=0)),'xy_overlap':xy,
        'vertical_gap_to_box_top_m':float(alo[2]-bhi[2]) if xy else None}


def support_candidates(boxes,object_position):
    p=np.asarray(object_position)
    return [b for b in boxes if np.all(p[:2]>=np.array(b['minimum_m'])[:2]) and np.all(p[:2]<=np.array(b['maximum_m'])[:2])
        and p[2]-.08<=b['maximum_m'][2]<=p[2]+.005]


class CollisionGeometry:
    def __init__(self,runtime):
        self.r=runtime;self.fingers=[];self.object_shapes=[];self.supports=[]
        self.model={'status':'unavailable','errors':[],
            'scope':'authored collision geometry AABBs plus live rigid-body poses; not solver contact shapes, exact fingertip distances or contact forces',
            'contact_forces':{'status':'unavailable','reason':'no contact sensor configured'},
            'usd_api_reference':'https://openusd.org/release/api/class_usd_geom_b_box_cache.html'}
        try:self._prepare()
        except Exception as error:self.model['errors'].append(type(error).__name__+': '+str(error))

    def _prepare(self):
        from pxr import Usd,UsdGeom,UsdPhysics
        from isaaclab.sim import get_current_stage
        stage=get_current_stage();cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],False,True)
        transforms=UsdGeom.XformCache(Usd.TimeCode.Default());prims=list(stage.Traverse());r=self.r
        robot_path=r.geometry['body_resolution']['robot_usd_path']
        def descendant(path,parent):return path==parent or path.startswith(parent+'/')
        def get_bound(prim,body=None):
            box=cache.ComputeWorldBound(prim) if body is None else (cache.ComputeUntransformedBound(prim) if prim==body else cache.ComputeRelativeBound(prim,body))
            bound=box.ComputeAlignedRange()
            if bound.IsEmpty():raise ValueError('empty authored bound '+str(prim.GetPath()))
            lo=list(bound.GetMin());hi=list(bound.GetMax());corners(lo,hi)
            return {'minimum_m':lo,'maximum_m':hi}
        def shapes(body):
            linear=np.array(transforms.GetLocalToWorldTransform(body),dtype=float)[:3,:3]
            scale=rigid_scale(linear)
            result=[];body_path=str(body.GetPath())
            for prim in prims:
                if not descendant(str(prim.GetPath()),body_path) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
                if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
                # A collider on another rigid child cannot use this body's live transform.
                ancestor=prim
                while ancestor and ancestor!=body and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):ancestor=ancestor.GetParent()
                if ancestor!=body:continue
                try:
                    bound=get_bound(prim,body)
                    result.append({'path':str(prim.GetPath()),'local_corners_m':(corners(bound['minimum_m'],bound['maximum_m'])*scale).tolist(),
                        'baked_body_scale':scale.tolist(),
                        'authored_offsets':{key:str(prim.GetAttribute(key).Get()) for key in ('physxCollision:contactOffset','physxCollision:restOffset') if prim.HasAttribute(key)}})
                except Exception as error:self.model['errors'].append(str(error))
            return result
        for name in ('panda_leftfinger','panda_rightfinger'):
            ids=[i for i,n in enumerate(r.robot.body_names) if n==name]
            bodies=[p for p in prims if p.GetName()==name and descendant(str(p.GetPath()),robot_path) and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(ids)!=1 or len(bodies)!=1:raise ValueError('finger rigid body is not uniquely resolved: '+name)
            found=shapes(bodies[0]);self.fingers.append({'name':name,'body_id':ids[0],'body_path':str(bodies[0].GetPath()),'shapes':found})
        pattern=str(r.base.scene['object'].cfg.prim_path).replace('{ENV_REGEX_NS}',robot_path.rsplit('/',1)[0])
        objects=[p for p in prims if re.fullmatch(pattern,str(p.GetPath())) and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(objects)==1:self.object_shapes=shapes(objects[0])
        else:self.model['errors'].append('object rigid root not uniquely resolved: '+pattern)
        static=[]
        for prim in prims:
            path=str(prim.GetPath())
            if descendant(path,robot_path) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
            ancestor=prim;dynamic=False
            while ancestor:
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(ancestor).GetRigidBodyEnabledAttr().Get() is not False:dynamic=True;break
                ancestor=ancestor.GetParent()
            if dynamic:continue
            try:static.append({'path':path,**get_bound(prim)})
            except Exception:continue
        self.supports=support_candidates(static,r.state()['object_pos_world_m'])
        self.model.update(status='available' if all(f['shapes'] for f in self.fingers) and self.object_shapes and self.supports else 'partial',
            fingers=self.fingers,object_shapes=self.object_shapes,static_support_candidates=self.supports,
            support_selection='static collider AABBs under initial object XY, box top within -80/+5mm of object root; no support identity assumed',
            transform_note='local bounds captured once with positive orthogonal body scale baked in; live physics poses used every sample; shear/reflection rejected; collision/drive properties unchanged')

    def sample(self):
        if self.model['status']=='unavailable':return {'status':'unavailable','reason':'; '.join(self.model['errors'])}
        try:
            r=self.r;t=r.tensor;obj=r.base.scene['object']
            object_position=t(obj.data.root_pos_w)[0].detach().cpu().tolist();object_quat=t(obj.data.root_quat_w)[0].detach().cpu().tolist()
            object_boxes=[{'path':x['path'],**live_bound(x['local_corners_m'],object_position,object_quat)} for x in self.object_shapes]
            fingers=[]
            for finger in self.fingers:
                index=finger['body_id'];position=t(r.robot.data.body_pos_w)[0,index].detach().cpu().tolist();quat=t(r.robot.data.body_quat_w)[0,index].detach().cpu().tolist()
                boxes=[{'path':x['path'],**live_bound(x['local_corners_m'],position,quat)} for x in finger['shapes']]
                pairs=[{'finger_shape':a['path'],'other_shape':b['path'],**box_relation(a,b)} for a in boxes for b in self.supports]
                object_pairs=[{'finger_shape':a['path'],'other_shape':b['path'],**box_relation(a,b)} for a in boxes for b in object_boxes]
                fingers.append({'name':finger['name'],'body_origin_world_m':position,'collision_bounds_world':boxes,
                    'support_pairs':pairs,'object_pairs':object_pairs})
            return {'status':self.model['status'],'fingers':fingers,'object_bounds_world':object_boxes,
                'contact_force_status':'unavailable','scope':'bounding-box proximity/overlap only; overlap is not proof of contact'}
        except Exception as error:return {'status':'unavailable','reason':type(error).__name__+': '+str(error)}


def geometry_summary(rows):
    available=[x for x in rows if x.get('collision_geometry',{}).get('status') in ('available','partial')]
    table=[p for x in available for f in x['collision_geometry']['fingers'] for p in f['support_pairs']]
    objects=[p for x in available for f in x['collision_geometry']['fingers'] for p in f['object_pairs']]
    deflected=next((x for x in rows if min(x['finger_target_m'])>=.039 and min(x['finger_position_m'])<.038),None)
    return {'sample_count':len(rows),'geometry_samples':len(available),
        'fully_available_samples':sum(x.get('collision_geometry',{}).get('status')=='available' for x in rows),
        'partial_samples':sum(x.get('collision_geometry',{}).get('status')=='partial' for x in rows),
        'unavailable_samples':len(rows)-len(available),
        'minimum_support_aabb_distance_m':min((x['aabb_distance_m'] for x in table),default=None),
        'minimum_object_aabb_distance_m':min((x['aabb_distance_m'] for x in objects),default=None),
        'first_open_target_deflection':None if deflected is None else {key:deflected[key] for key in ('control_index','physics_substep','finger_target_m','finger_position_m','tcp_world_m','collision_geometry')},
        'scope':'2mm finger-target discrepancy is a diagnostic marker; AABBs do not measure contacts or forces'}
