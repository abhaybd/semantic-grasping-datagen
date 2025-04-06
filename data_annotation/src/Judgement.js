import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router';
import ObjectViewer from './ObjectViewer';
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

  useEffect(() => {
    const fetchAnnotation = async () => {
      setLoading(true);
      const category = searchParams.get('category');
      const objectId = searchParams.get('object_id');
      const graspId = searchParams.get('grasp_id');
      const userId = searchParams.get('user_id');
      const studyId = searchParams.get('study_id') || '';

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

    fetchAnnotation();
  }, [searchParams, navigate]);

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
    
    const category = searchParams.get('category');
    const objectId = searchParams.get('object_id');
    const graspId = searchParams.get('grasp_id');
    const userId = searchParams.get('user_id');
    const studyId = searchParams.get('study_id') || '';
    const judgerUserId = searchParams.get('judger_id') || 'anonymous';

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

      alert('Judgement submitted successfully!');
      navigate('/');
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