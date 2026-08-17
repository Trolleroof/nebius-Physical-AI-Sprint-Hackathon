"use client";

/**
 * The actual SO-101 follower arm, assembled from the official URDF.
 *
 * Meshes and joint origins come from TheRobotStudio/SO-ARM100 (Apache-2.0),
 * so this is the same geometry as the arm on the table rather than a stock
 * humanoid. It idles rather than mirroring live joint data — feeding it real
 * telemetry is a later upgrade, and `setJointValue` is where that would go.
 */

import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import URDFLoader, { type URDFRobot } from "urdf-loader";

const URDF_URL = "/robot/so101_new_calib.urdf";

/**
 * wrist_roll's motor is broken on the physical arm -- the joint is taped at
 * -pi/2 and never moves. The viewer holds it there and gives it no idle drift,
 * so what renders is a pose the real robot can actually be in.
 */
const WRIST_ROLL_LOCK = -Math.PI / 2;

/** Resting pose, in radians, keyed by URDF joint name. */
const IDLE_POSE: Record<string, number> = {
  shoulder_pan: 0.0,
  shoulder_lift: -0.55,
  elbow_flex: 1.15,
  wrist_flex: 0.55,
  wrist_roll: WRIST_ROLL_LOCK,
  gripper: 0.25,
};

/** How far each joint drifts from the resting pose, and how fast. */
const IDLE_MOTION: Record<string, { amp: number; freq: number; phase: number }> = {
  shoulder_pan: { amp: 0.22, freq: 0.22, phase: 0 },
  shoulder_lift: { amp: 0.1, freq: 0.3, phase: 1.1 },
  elbow_flex: { amp: 0.13, freq: 0.26, phase: 2.2 },
  wrist_flex: { amp: 0.1, freq: 0.34, phase: 0.6 },
  wrist_roll: { amp: 0, freq: 0, phase: 0 },
  gripper: { amp: 0.18, freq: 0.42, phase: 1.7 },
};

function Arm({ onReady }: { onReady: () => void }) {
  const [robot, setRobot] = useState<URDFRobot | null>(null);
  const group = useRef<THREE.Group>(null);

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: "#dfe4ea",
        roughness: 0.42,
        metalness: 0.18,
      }),
    [],
  );

  useEffect(() => {
    let cancelled = false;
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);

    // urdf-loader hands us each mesh path and the material it parsed from the
    // URDF; we ignore that material and use one house material so the whole
    // arm reads as a single object. Every mesh in this model is STL.
    loader.loadMeshCb = (path, mgr, _urdfMaterial, done) => {
      new STLLoader(mgr).load(
        path,
        (geometry) => {
          geometry.computeVertexNormals();
          const mesh = new THREE.Mesh(geometry, material);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          done(mesh);
        },
        undefined,
        // A single missing link should cost us that link, not the whole arm.
        (err) => done(new THREE.Object3D(), err as Error),
      );
    };

    loader.load(URDF_URL, (loaded) => {
      if (cancelled) return;

      // URDF is Z-up, three.js is Y-up.
      loaded.rotation.x = -Math.PI / 2;

      Object.entries(IDLE_POSE).forEach(([joint, value]) => {
        if (loaded.joints[joint]) loaded.setJointValue(joint, value);
      });

      // Frame the arm: centre on its own bounds, then normalise scale so the
      // camera framing does not depend on the model's real-world units.
      loaded.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(loaded);
      const size = box.getSize(new THREE.Vector3());
      const centre = box.getCenter(new THREE.Vector3());
      const scale = 2.4 / Math.max(size.x, size.y, size.z);

      // Scale applies to children, position to the parent frame, so the
      // centring offset has to be scaled too.
      loaded.scale.setScalar(scale);
      loaded.position.set(-centre.x * scale, -centre.y * scale - 0.15, -centre.z * scale);

      setRobot(loaded);
      onReady();
    });

    return () => {
      cancelled = true;
    };
  }, [material, onReady]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (group.current) group.current.rotation.y = Math.sin(t * 0.12) * 0.55 - 0.25;
    if (!robot) return;
    for (const [joint, motion] of Object.entries(IDLE_MOTION)) {
      if (!robot.joints[joint]) continue;
      const base = IDLE_POSE[joint] ?? 0;
      robot.setJointValue(joint, base + Math.sin(t * motion.freq + motion.phase) * motion.amp);
    }
  });

  return <group ref={group}>{robot && <primitive object={robot} />}</group>;
}

export function Robot({ className = "" }: { className?: string }) {
  const [ready, setReady] = useState(false);

  return (
    <div className={`relative ${className}`}>
      <Canvas
        camera={{ position: [2.6, 1.5, 3.0], fov: 38 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={0.55} />
        <directionalLight position={[4, 6, 3]} intensity={2.1} />
        {/* Blue rim light: the only colour on the model, matching the HUD accent. */}
        <directionalLight position={[-5, 2, -4]} intensity={2.4} color="#2e8bff" />
        <directionalLight position={[0, -3, 2]} intensity={0.35} color="#8fb8ff" />
        <Arm onReady={() => setReady(true)} />
      </Canvas>

      {!ready && (
        <div className="absolute inset-0 grid place-items-center">
          <span className="label blink">Loading geometry…</span>
        </div>
      )}
    </div>
  );
}
