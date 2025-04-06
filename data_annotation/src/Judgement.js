import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router';
import ObjectViewer from './ObjectViewer';
import ProgressBar from './ProgressBar';
import './DataAnnotation.css';
import './Judgement.css';

const Judgement = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [annotation, setAnnotation] = useState(null);
  const [judgement, setJudgement] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [startTime, setStartTime] = useState(Date.now());
  const [objectLoading, setObjectLoading] = useState(true);
  const [judgementSchedule, setJudgementSchedule] = useState(null);

  const encodeStr = (str) => {
    return encodeURIComponent(btoa(str));
  };

  const decodeStr = (str) => {
    return atob(decodeURIComponent(str));
  };

  const navigateToSchedule = (schedule) => {
    const newParams = new URLSearchParams(searchParams);
    newParams.set("judgement_schedule", encodeStr(JSON.stringify(schedule)));
    navigate({
      pathname: "/judgement",
      search: newParams.toString()
    }, {replace: true});
  };

  useEffect(() => {
    if (searchParams.has("judgement_schedule")) {
      const schedule = JSON.parse(decodeStr(searchParams.get("judgement_schedule")));
      const idx = schedule.idx;
      if (idx >= schedule.judgements.length || idx < 0) {
        alert("Invalid schedule index!");
        navigate('/');
        return;
      }
      setJudgementSchedule(schedule);
      
      // Set URL params from the current judgement in the schedule
      const currentJudgement = schedule.judgements[idx];
      fetchAnnotation(
        currentJudgement.category,
        currentJudgement.object_id,
        currentJudgement.grasp_id,
        currentJudgement.user_id,
        currentJudgement.study_id || ''
      );
    } else {
      alert('Missing judgement schedule!');
      navigate('/');
    }
  }, [searchParams, navigate]);

  const fetchAnnotation = async (category, objectId, graspId, userId, studyId) => {
    setLoading(true);
    setJudgement('');

    if (!category || !objectId || !graspId || !userId) {
      alert('Missing required parameters!');
      navigate('/');
      return;
    }

    try {
      const response = await fetch(`/api/get-annotation/${category}/${objectId}/${graspId}/${userId}?study_id=${studyId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch annotation: HTTP ${response.status}`);
      }
      
      const data = await response.json();
      setAnnotation(data);
      setStartTime(Date.now());
      setLoading(false);
    } catch (error) {
      console.error('Error fetching annotation:', error);
      alert('Error fetching annotation: ' + error.message);
      setLoading(false);
    }
  };

  const handleJudgementSelect = (value) => {
    setJudgement(value);
  };

  const handleSubmit = async () => {
    if (!judgement) {
      alert('Please select a judgement!');
      return;
    }

    setSubmitting(true);
    const timeTaken = (Date.now() - startTime) / 1000;
    
    let category, objectId, graspId, userId, studyId, judgerUserId;
    
    if (judgementSchedule) {
      const currentJudgement = judgementSchedule.judgements[judgementSchedule.idx];
      category = currentJudgement.category;
      objectId = currentJudgement.object_id;
      graspId = currentJudgement.grasp_id;
      userId = currentJudgement.user_id;
      studyId = currentJudgement.study_id || '';
      judgerUserId = currentJudgement.judger_id || searchParams.get('judger_id') || 'anonymous';
    } else {
      category = searchParams.get('category');
      objectId = searchParams.get('object_id');
      graspId = searchParams.get('grasp_id');
      userId = searchParams.get('user_id');
      studyId = searchParams.get('study_id') || '';
      judgerUserId = searchParams.get('judger_id') || 'anonymous';
    }

    const annotKey = `${studyId}__${category}__${objectId}__${graspId}__${userId}`;

    try {
      const response = await fetch('/api/submit-judgement', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          annot_key: annotKey,
          judgement_label: judgement,
          user_id: judgerUserId,
          time_taken: timeTaken
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to submit judgement: HTTP ${response.status}`);
      }

      if (judgementSchedule) {
        if (judgementSchedule.idx + 1 === judgementSchedule.judgements.length) {
          alert('All judgements completed!');
          navigate('/');
        } else {
          const newSchedule = { ...judgementSchedule, idx: judgementSchedule.idx + 1 };
          navigateToSchedule(newSchedule);
        }
      } else {
        alert('Judgement submitted successfully!');
        navigate('/');
      }
    } catch (error) {
      console.error('Error submitting judgement:', error);
      alert('Error submitting judgement: ' + error.message);
    } finally {
      setSubmitting(false);
    }
  };

  const isFullyLoaded = !loading && !objectLoading;

  return (
    <div className="data-annotation-container">
      <h2>Judgement Page</h2>
      
      {judgementSchedule && judgementSchedule.judgements.length > 1 && (
        <div className="progress-container">
          <span>{judgementSchedule.idx + 1}/{judgementSchedule.judgements.length}</span>
          <ProgressBar completed={judgementSchedule.idx} total={judgementSchedule.judgements.length} />
        </div>
      )}
      
      <div className="content-container">
        <div className="object-viewer-container">
          {annotation && (
            <ObjectViewer
              object_category={annotation.obj.object_category}
              object_id={annotation.obj.object_id}
              grasp_id={annotation.grasp_id}
              onFinishedLoading={() => setObjectLoading(false)}
            />
          )}
          {objectLoading && <div className="loading-overlay">Loading 3D model...</div>}
        </div>
        
        <div className="annotation-form-container">
          {loading ? (
            <div className="loading">Loading annotation data...</div>
          ) : annotation ? (
            <div className="judgement-container">
              <h3>Annotation Details</h3>
              
              <div className="annotation-details">
                <div className="detail-row">
                  <strong>Object Description:</strong>
                  <p>{annotation.obj_description}</p>
                </div>
                
                <div className="detail-row">
                  <strong>Grasp Description:</strong>
                  <p>{annotation.grasp_description}</p>
                </div>
                
                <div className="detail-row">
                  <strong>Grasp Label:</strong>
                  <p>{annotation.grasp_label}</p>
                </div>
              </div>
              
              <div className="judgement-selection">
                <h3>Your Judgement</h3>
                <div className="judgement-options">
                  <button 
                    className={`judgement-button ${judgement === 'good' ? 'selected' : ''}`}
                    onClick={() => handleJudgementSelect('good')}
                    disabled={!isFullyLoaded}
                  >
                    Good
                  </button>
                  
                  <button 
                    className={`judgement-button ${judgement === 'mid' ? 'selected' : ''}`}
                    onClick={() => handleJudgementSelect('mid')}
                    disabled={!isFullyLoaded}
                  >
                    Mid
                  </button>
                  
                  <button 
                    className={`judgement-button ${judgement === 'bad' ? 'selected' : ''}`}
                    onClick={() => handleJudgementSelect('bad')}
                    disabled={!isFullyLoaded}
                  >
                    Bad
                  </button>
                </div>
                
                <button 
                  className="ai2-button submit-button"
                  onClick={handleSubmit}
                  disabled={!judgement || submitting || !isFullyLoaded}
                >
                  {submitting ? 'Submitting...' : 'Submit Judgement'}
                </button>
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