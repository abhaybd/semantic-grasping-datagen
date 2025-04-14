import React from 'react';
import { FaTimes } from 'react-icons/fa';
import teapot_img from "./tutorial_teapot.png";
import './JudgementTutorial.css';

const JudgementTutorial = ({ onClose }) => {
  return (
    <div className="tutorial-overlay">
      <div className="tutorial-popup">
        <button className="close-button" onClick={onClose}>
          <FaTimes />
        </button>
        <div className="tutorial-content">
          <h2>Grasp Description Guidelines</h2>

          <p>
            This is an interface for you to judge the quality of existing grasp descriptions.
            You will be given a 3D model of an object, along with a proposed way to grasp it.
            Given a description of how the object is being grasped, your job is to judge whether it is an accurate description of the grasp.
            The grasp is represented by the green "pitchfork", showing the position and orientation of the robot's parallel-jaw gripper.
          </p>
          <p>
            If the description is inaccurate, you should provide a corrected description of how the object is being grasped.
            In general, you should adhere to the style of the provided description, and only correct factual errors.
          </p>
          <p>
            In extreme cases, you may not know the correct grasp description, and you should mark the description as unsure.
            Please minimize the number of unsure judgements, but it is preferable to mark unsure than to guess.
          </p>
          
          <h3>What Makes a Good Grasp Description</h3>
          <p>
            A good grasp description should be:
          </p>
          <ul>
            <li><strong>Specific:</strong> Clearly identify the exact location of the grasp on the object</li>
            <li><strong>Detailed:</strong> Include information about where the fingers are positioned</li>
            <li><strong>Objective:</strong> Focus on describing the grasp itself, not its quality or effectiveness</li>
            <li><strong>Concise:</strong> Use clear, direct language without unnecessary details</li>
          </ul>

          <h3>Example of a Good Description</h3>
          <img src={teapot_img} alt="Teapot example" className="tutorial-image" />
          <blockquote>
            "The grasp is on the spout of the teapot, where it connects to the body. The grasp is oriented parallel to the base of the teapot, and the fingers are closing on either side of the spout."
          </blockquote>
          <p>
            This description is good because it:
          </p>
          <ul>
            <li>Specifically identifies the location (spout where it connects to the body)</li>
            <li>Explains how the fingers are positioned (closing on either side)</li>
            <li>Focuses only on describing the grasp itself</li>
          </ul>

          <h3>What to Avoid</h3>
          <ul>
            <li>Don't include subjective judgments (e.g., "This is a good grasp" or "This grasp would be unstable")</li>
            <li>Don't suggest alternative grasps or improvements</li>
            <li>Don't describe what would happen after the grasp (e.g., "The object would fall")</li>
            <li>Don't use vague terms like "somewhere" or "around"</li>
          </ul>

          <h3>Example of a Poor Description</h3>
          <blockquote>
            "This is a bad grasp because it's on the wrong part of the object. It would be better to grasp the handle instead."
          </blockquote>
        </div>
      </div>
    </div>
  );
};

export default JudgementTutorial; 