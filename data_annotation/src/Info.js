import React from 'react';

function Info() {
  return (
    <div className="info-body">
      <h2>Build Information</h2>
      <div className="info-content">
        <p><tt><strong>Git Commit:</strong> {process.env.REACT_APP_GIT_COMMIT || 'Not available'}</tt></p>
        <p><tt><strong>Build Timestamp:</strong> {process.env.REACT_APP_TIMESTAMP || 'Not available'}</tt></p>
      </div>
    </div>
  );
}

export default Info; 