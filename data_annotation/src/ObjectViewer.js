import React, { useState, useEffect } from 'react';
import { Canvas, useLoader } from '@react-three/fiber';
import { Environment, OrbitControls } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { Suspense } from 'react';
import "./ObjectViewer.css";


const ObjectViewer = ({ object_category, object_id, grasp_id, onFinishedLoading }) => {
  const [orbitControls, setOrbitControls] = useState(null);

  const GLTFMesh = ({ meshURL }) => {
    const gltf = useLoader(GLTFLoader, meshURL);
    useEffect(() => {
      if (onFinishedLoading) {
        onFinishedLoading();
      }
    }, [gltf]);
    return <primitive object={gltf.scene} />;
  };

  useEffect(() => {
    const arr = [!!object_category, !!object_id, grasp_id != null];
    if (arr.some(Boolean) && !arr.every(Boolean)) {
      throw new Error(`Incomplete object data: category=${object_category}, id=${object_id}, grasp_id=${grasp_id}`);
    }
  }, [object_category, object_id, grasp_id]);

  const hasData = !!object_category && !!object_id && grasp_id != null;

  const Spinner = () => <div className="spinner"></div>;

  return (
    <div className="canvas-container-toolbar">
      <div className="canvas-container">
        {hasData && (
            <Suspense fallback={<Spinner />}>
            <Canvas camera={{ position: [0, 0.4, 0.6], near: 0.05, far: 20, fov: 45 }}>
              <Environment preset="sunset" />
              <OrbitControls ref={setOrbitControls} />
              <GLTFMesh
                meshURL={`/api/get-mesh-data/${object_category}/${object_id}/${grasp_id}`}
              />
            </Canvas>
          </Suspense>
        )}
      </div>
      <div className="canvas-toolbar">
        <p className='instructions'>Left click + drag to rotate, right click + drag to pan, scroll to zoom.</p>
        <button onClick={() => orbitControls.reset()} className="ai2-button" disabled={!orbitControls}>Reset View</button>
      </div>
    </div>
  );
};

export default ObjectViewer;
