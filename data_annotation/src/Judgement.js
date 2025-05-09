import React, { useEffect, useState } from "react";
import { useSearchParams, useNavigate } from "react-router";
import ObjectViewer from "./ObjectViewer";
import ProgressBar from "./ProgressBar";
import JudgementTutorial from "./JudgementTutorial";
import "./DataAnnotation.css";
import "./Judgement.css";
import { FaCheckCircle, FaQuestionCircle, FaTimesCircle } from 'react-icons/fa';
import { BsBoxArrowUpRight } from 'react-icons/bs';

/*
Judgement schedule looks like:
{
  idx: int,
  judgements: [
    {
      object_category: str,
      object_id: str,
      grasp_id: int,
      user_id: str,
      study_id: str
    },
    ...
  ]
}
*/

const Judgement = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [annotation, setAnnotation] = useState(null);
  const [judgement, setJudgement] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [startTime, setStartTime] = useState(Date.now());
  const [objectLoading, setObjectLoading] = useState(true);
  const [judgementSchedule, setJudgementSchedule] = useState(null);
  const [judgerUserId, setJudgerUserId] = useState("");
  const [showTutorial, setShowTutorial] = useState(false);
  const [correctGraspDescription, setCorrectGraspDescription] = useState("");

  const encodeStr = (str) => {
    return encodeURIComponent(btoa(str));
  };

  const decodeStr = (str) => {
    return atob(decodeURIComponent(str));
  };

  const navigateToSchedule = (schedule) => {
    const newParams = new URLSearchParams(searchParams);
    newParams.set("judgement_schedule", encodeStr(JSON.stringify(schedule)));
    navigate(
      {
        pathname: "/judgement",
        search: newParams.toString(),
      },
      { replace: true }
    );
  };

  useEffect(() => {
    if (searchParams.has("judgement_schedule")) {
      const schedule = JSON.parse(
        decodeStr(searchParams.get("judgement_schedule"))
      );
      const idx = schedule.idx;
      if (idx >= schedule.judgements.length || idx < 0) {
        throw new Error(`Invalid schedule index! idx=${idx}, length=${schedule.judgements.length}`);
      }
      setJudgementSchedule(schedule);

      const currentJudgement = schedule.judgements[idx];

      fetchAnnotation(
        currentJudgement.object_category,
        currentJudgement.object_id,
        currentJudgement.grasp_id,
        currentJudgement.user_id,
        currentJudgement.study_id
      );
    } else {
      throw new Error("Missing judgement schedule!");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams, navigate]);

  useEffect(() => {
    if (annotation) {
      setCorrectGraspDescription(annotation.grasp_description);
    }
  }, [annotation]);

  useEffect(() => {
    const hasSeenTutorial = localStorage.getItem('hasSeenJudgementTutorial');
    if (!hasSeenTutorial) {
      setShowTutorial(true);
      localStorage.setItem('hasSeenJudgementTutorial', 'true');
    }
  }, []);

  const fetchAnnotation = async (
    category,
    objectId,
    graspId,
    userId,
    studyId
  ) => {
    setLoading(true);
    setJudgement("");

    if (!category || !objectId || !graspId || !userId) {
      throw new Error(`Missing required parameters! category=${category}, objectId=${objectId}, graspId=${graspId}, userId=${userId}`);
    }

    try {
      const response = await fetch(
        `/api/get-annotation/${category}/${objectId}/${graspId}/${userId}?study_id=${studyId}&synthetic=true`
      );

      if (!response.ok) {
        if (response.status === 404) {
          const data = await response.json();
          throw new Error(data.detail);
        } else {
          throw new Error(`Error fetching annotation: HTTP ${response.status}`);
        }
      }

      const data = await response.json();
      setAnnotation(data);
      setStartTime(Date.now());
      setLoading(false);
    } catch (error) {
      console.error("Error fetching annotation:", error);
      throw error;
    }
  };

  const handleSubmit = async () => {
    if (!judgement) {
      alert("Please select a judgement!");
      return;
    }

    setSubmitting(true);
    const timeTaken = (Date.now() - startTime) / 1000;
    const studyId = searchParams.get("study_id") || "";
    try {
      const response = await fetch("/api/submit-judgement", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          annotation: annotation,
          judgement_label: judgement,
          user_id: judgerUserId,
          time_taken: timeTaken,
          study_id: studyId,
          correct_grasp_description: judgement === "inaccurate" ? correctGraspDescription : null,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to submit judgement: HTTP ${response.status}`);
      }

      if (judgementSchedule.idx + 1 === judgementSchedule.judgements.length) {
        if (searchParams.has("prolific_code")) {
          window.location.href = `https://app.prolific.com/submissions/complete?cc=${searchParams.get("prolific_code")}`;
        } else {
          navigate('/done', { replace: true });
        }
      } else {
        const newSchedule = {
          ...judgementSchedule,
          idx: judgementSchedule.idx + 1,
        };
        navigateToSchedule(newSchedule);
      }
    } catch (error) {
      console.error("Error submitting judgement:", error);
      alert("Error submitting judgement: " + error.message);
    } finally {
      setSubmitting(false);
    }
  };

  const isFullyLoaded = !loading && !objectLoading;

  return (
    <div className="data-annotation-container">
      <div className="header-buttons">
        <button className="ai2-button" onClick={() => setShowTutorial(true)}>
          Show Tutorial
        </button>
        <button className="ai2-button" onClick={() => window.open("/reference", '_blank')}>
          Show Examples
          <BsBoxArrowUpRight style={{marginLeft: "5px"}} />
        </button>
      </div>

      {showTutorial && <JudgementTutorial onClose={() => setShowTutorial(false)} />}

      {judgementSchedule && judgementSchedule.judgements.length > 1 && (
        <div className="progress-container">
          <span>
            {judgementSchedule.idx + 1}/{judgementSchedule.judgements.length}
          </span>
          <ProgressBar
            completed={judgementSchedule.idx}
            total={judgementSchedule.judgements.length}
          />
        </div>
      )}

      <div className="content-container" style={{alignItems: "flex-start"}}>
        <div className="object-viewer-container">
          {annotation && (
            <ObjectViewer
              object_category={annotation.obj.object_category}
              object_id={annotation.obj.object_id}
              grasp_id={annotation.grasp_id}
              onFinishedLoading={() => setObjectLoading(false)}
            />
          )}
          {objectLoading && (
            <div className="loading-overlay">Loading 3D model...</div>
          )}
        </div>

        <div className="annotation-form-container">
          {loading ? (
            <div className="loading">Loading annotation data...</div>
          ) : annotation ? (
            <div className="judgement-container">
              <div>
                <h3 className="form-subtitle">Annotation Details</h3>
                <div className="annotation-details">
                  <div>
                    <strong>Object Category:</strong> {annotation.obj.object_category}
                  </div>
                  <br />
                  <div className="detail-row">
                    <strong>Grasp Description:</strong>
                    <p>{annotation.grasp_description}</p>
                  </div>
                </div>
              </div>

              <div className="judgement-selection">
                <h3 className="form-subtitle">Annotation Judgement</h3>
                <div className="judgement-form-container">
                  <div className="user-id-container">
                    <label htmlFor="judgerUserId">User ID:</label>
                    <input
                      type="text"
                      id="judgerUserId"
                      value={judgerUserId}
                      onChange={e => setJudgerUserId(e.target.value)}
                      placeholder="Enter your user ID"
                      className="user-id-input"
                    />
                  </div>
                  <div className="judgement-options">
                    <button
                      className={`judgement-button ${judgement === "accurate" ? "selected" : ""}`}
                      onClick={() => {
                        setJudgement("accurate");
                      }}
                      disabled={!isFullyLoaded}
                      data-judgement="accurate"
                    >
                      <FaCheckCircle className="judgement-icon" />
                      Accurate
                    </button>

                    <button
                      className={`judgement-button ${judgement === "uncertain" ? "selected" : ""}`}
                      onClick={() => {
                        setJudgement("uncertain");
                      }}
                      disabled={!isFullyLoaded}
                      data-judgement="uncertain"
                    >
                      <FaQuestionCircle className="judgement-icon" />
                      Unsure
                    </button>

                    <button
                      className={`judgement-button ${judgement === "inaccurate" ? "selected" : ""}`}
                      onClick={() => setJudgement("inaccurate")}
                      disabled={!isFullyLoaded}
                      data-judgement="inaccurate"
                    >
                      <FaTimesCircle className="judgement-icon" />
                      Inaccurate
                    </button>
                  </div>

                  <div className={`correct-grasp-container ${judgement === "inaccurate" ? "visible" : ""}`}>
                    <label htmlFor="correctGraspDescription">Corrected Grasp Description:</label>
                    <textarea
                      id="correctGraspDescription"
                      value={correctGraspDescription}
                      onChange={e => setCorrectGraspDescription(e.target.value)}
                      placeholder="Describe how the object should be grasped..."
                      className="correct-grasp-textarea"
                    />
                  </div>

                  <button
                    className="ai2-button submit-button"
                    onClick={handleSubmit}
                    disabled={!judgement || submitting || !isFullyLoaded || !judgerUserId.trim() || (judgement === "inaccurate" && (!correctGraspDescription.trim() || correctGraspDescription === annotation.grasp_description))}
                  >
                    {submitting ? "Submitting..." : "Submit Judgement"}
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="error-message">Failed to load annotation data</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Judgement;
