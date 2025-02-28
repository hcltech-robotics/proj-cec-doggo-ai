import omni.replicator.core as rep
import omni.usd as usd
import datetime
import asyncio
import json
import os
from typing import Union
from pxr import Usd, Sdf
import random


rep.set_global_seed(random.randint(0, 1000000))

ENV_PATH='omniverse://localhost/Library/jetty_with_gauge.usdc'
GAUGE_PATH="/Replicator/Ref_Xform/Ref/Pressure_Gauge_2/Pressure_Gauge/Pressure_Gauge_uw_obj"
GAUGE_NEEDLE_PATH=f"/Replicator/Ref_Xform/Ref/Pressure_Gauge_2/Pressure_Gauge/Pressure_Gauge_uw_obj/Metal/Hand"
LIGHT_PATH =f"/Replicator/Ref_Xform/Ref/Environment/sky/DomeLight"
ROTATION_MIN = -134
ROTATION_MAX = 137
CAMERA_PATH="/Replicator/Ref_Xform/Ref/RepCamera"

NOW=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
OUTPUT_DIR = os.path.expanduser("~/data/gauge_data_basic_"+NOW)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_rotation(rotation, min, max):
    return 1 - (rotation - min) / (max - min)

def get_current_stage() -> Usd.Stage:
    return usd.get_context().get_stage()

def get_attribute_value(prim: Usd.Prim, attribute_name: str):
    attr = prim.GetAttribute(attribute_name)
    return attr.Get()

def get_prim_by_path(stage: Usd.Stage, prim_path: Union[str, Sdf.Path]) -> Usd.Prim:
    return stage.GetPrimAtPath(prim_path)

def uniform_random_rotation(prim, min_x = 0, min_y = 0, min_z = 0, max_x = 0, max_y = 0, max_z = 0):
    with prim:
        rotation = rep.distribution.uniform((min_x, min_y, min_z), (max_x, max_y, max_z))
        rep.modify.pose(rotation = rotation)


with rep.new_layer() as layer:
    stage = usd.get_context().get_stage()
    #rep.settings.carb_settings("/omni/replicator/RTSubframes", 20)
    ground = rep.create.from_usd(ENV_PATH)
    gauge = rep.get.prim_at_path(GAUGE_PATH)
    camera =  rep.get.camera(CAMERA_PATH)
    light = rep.get.light(LIGHT_PATH)
    camera_prim = get_prim_by_path(stage, CAMERA_PATH)
    gauge_prim = get_prim_by_path(stage, GAUGE_PATH)
    gauge_location = get_attribute_value(gauge_prim, "xformOp:translate")
    camera_location = get_attribute_value(camera_prim, "xformOp:translate")
    print(gauge_location)
    print(camera_location)
    # with camera:
    #     rep.modify.pose(look_at = gauge_location)

    render_product = rep.create.render_product(camera, resolution=(1920, 1080))
    
    with gauge:
      rep.modify.semantics([('class', "gauge")])
    	
    hand = rep.get.prim_at_path(GAUGE_NEEDLE_PATH)
    with hand:
        rep.modify.semantics([('class', "gauge_needle")])

    rep.randomizer.register(uniform_random_rotation)
    with rep.trigger.on_frame():

        uniform_random_rotation(hand, min_z = ROTATION_MIN, max_z = ROTATION_MAX)
        with gauge:
            rep.modify.pose(rotation = rep.distribution.uniform((0, -60, 0), (0, 60, 0)))

        with camera:
             rep.modify.pose(position=rep.distribution.uniform((camera_location[0]-0.1, camera_location[1]-0.2, camera_location[2]-0.2)
                                                             , (camera_location[0]+0.1, camera_location[1]+0.2, camera_location[2]+0.2)))
        with light:
            rep.modify.attribute("intensity", rep.distribution.uniform(0.0, 1.2))
            rep.modify.attribute("exposure", rep.distribution.uniform(0.0, 1.5))
            rep.modify.attribute("color", rep.distribution.uniform((0.1, 0.1, 0.1), (1.0, 1.0, 1.0)))
        
    
    
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize( output_dir=OUTPUT_DIR, rgb=True, bounding_box_2d_tight=True, semantic_types=["class"])
    writer.attach([render_product])
    

async def run():

    rotations = []
    rotations_filename = "rotations.json"
    stage = get_current_stage()
    for i in range(0, 5000):
        # This renders one new frame (the subframes are needed for high quality raytracing)
        await rep.orchestrator.step_async(rt_subframes=5)
        
        # Access a prim when the simulation is not running and read it's rotation.
        prim = get_prim_by_path(stage, GAUGE_NEEDLE_PATH)
        rotation = get_attribute_value(prim, "xformOp:rotateXYZ")
        rotation_norm = normalize_rotation(rotation[2], ROTATION_MIN, ROTATION_MAX)
        rotations.append({"frame" : i, "rotation" : rotation_norm})
        print(f"Step {i}, rotation: {rotation} rotation_norm: {rotation_norm}")

    # After the render we write the rotations to file
    with open(os.path.join(OUTPUT_DIR, rotations_filename), "w") as f:
        json.dump(rotations, f, indent=2)

asyncio.ensure_future(run())