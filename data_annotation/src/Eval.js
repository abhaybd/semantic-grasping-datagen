import React, { useState, useRef } from 'react';
import { Canvas, useLoader } from '@react-three/fiber';
import { Environment, OrbitControls } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { Suspense } from 'react';
import './Eval.css';

const SceneViewer = ({ sceneId, orbitControlsRef, renderKey }) => {
  const GLTFMesh = ({ meshURL }) => {
    const gltf = useLoader(GLTFLoader, meshURL);
    return <primitive object={gltf.scene} />;
  };

  const Spinner = () => <div className="spinner"></div>;

  return (
    <div className="canvas-container-toolbar">
      <div className="canvas-container">
        {sceneId && (
          <Suspense fallback={<Spinner />}>
            <Canvas camera={{ position: [0, 0.4, 0.6], near: 0.05, far: 20, fov: 45 }}>
              <Environment preset="sunset" />
              <OrbitControls ref={orbitControlsRef} />
              <GLTFMesh
                meshURL={`/api/get-scene/${sceneId}/${renderKey}`}
              />
            </Canvas>
          </Suspense>
        )}
      </div>
      <div className="canvas-toolbar">
        <p className='instructions'>Left click + drag to rotate, right click + drag to pan, scroll to zoom.</p>
        <button onClick={() => orbitControlsRef.current?.reset()} className="ai2-button" disabled={!orbitControlsRef.current}>Reset View</button>
      </div>
    </div>
  );
};

const Eval = () => {
  const [sceneId, setSceneId] = useState(null);
  const [renderKey, setRenderKey] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const textInputRef = useRef(null);
  const orbitControlsRef = useRef(null);

  const generateScene = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/generate-scene/100', {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate scene');
      }
      
      const newSceneId = JSON.parse(await response.text());
      setSceneId(newSceneId);
      setRenderKey(0);
    } catch (err) {
      setError('Failed to generate scene: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!sceneId || !textInputRef.current.value) return;

    try {
      setLoading(true);
      setError(null);

      // Get current camera state from SceneViewer
      const cameraState = orbitControlsRef.current?.object;
      if (!cameraState) {
        throw new Error('Camera state not available');
      }

      const camPos = [cameraState.position.x, cameraState.position.y, cameraState.position.z];
      const camQuat = [cameraState.quaternion.x, cameraState.quaternion.y, cameraState.quaternion.z, cameraState.quaternion.w];

      const camParams = [cameraState.fov, 320, 240]; // Example values - adjust as needed

      const predictResponse = await fetch(`/api/predict/${sceneId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: textInputRef.current.value,
          cam_pos: camPos,
          cam_quat: camQuat,
          cam_params: camParams,
        }),
      });

      if (!predictResponse.ok) {
        throw new Error('Prediction failed');
      }

      setRenderKey(k => k + 1);
    } catch (err) {
      setError('Prediction failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="eval-container">
      <div className="controls">
        <button 
          onClick={generateScene} 
          disabled={loading}
          className="generate-button"
        >
          Generate Scene
        </button>

        <form onSubmit={handlePredict} className="predict-form">
          <input
            type="text"
            ref={textInputRef}
            placeholder="Enter grasp description..."
            disabled={!sceneId || loading}
          />
          <button 
            type="submit"
            disabled={!sceneId || loading}
          >
            Predict Grasps
          </button>
        </form>

        {error && <div className="error">{error}</div>}
      </div>

      <div className="viewer">
        {sceneId && (
          <SceneViewer
            sceneId={sceneId}
            orbitControlsRef={orbitControlsRef}
            renderKey={renderKey}
          />
        )}
      </div>
    </div>
  );
};

export default Eval;
