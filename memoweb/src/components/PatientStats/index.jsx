import './styles.css';
import { useState, useEffect } from 'react';

const PatientStats = ({ onMRIClick, onImageClick , apiBaseUrl }) => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/user_patients`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to fetch patients');
        }
        
        const data = await response.json();
        setPatients(data.patients || []);
        
        if (data.patients?.length > 0) {
          setSelectedPatient(data.patients[0]);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPatients();
  }, [apiBaseUrl]);

  const getStageDescription = (stage) => {
    switch(stage) {
      case "0": return "No Dementia";
      case "1": return "Very Mild Dementia";
      case "2": return "Mild Dementia";
      case "3": return "Moderate Dementia";
      default: return "Stage not assessed";
    }
  };
  if (loading) return <div>Loading patient data...</div>;
  if (error) return <div>Error: {error}</div>;
  if (patients.length === 0) return <div>No patients found</div>;


  return (
    <div className="patient-stats-container">
      {/* Patient selection dropdown */}
      <div className="patient-selector">
        <label htmlFor="patient-select">Select Patient:</label>
        <select
          id="patient-select"
          value={selectedPatient?._id || ''}
          onChange={(e) => {
            const patient = patients.find(p => p._id === e.target.value);
            setSelectedPatient(patient);
          }}
        >
          {patients.map(patient => (
            <option key={patient._id} value={patient._id}>
              {patient.name} (ID: {patient.patient_id})
            </option>
          ))}
        </select>
      </div>

      {/* Selected patient details */}
      {selectedPatient && (
        <div className="patient-details">
          <h3>{selectedPatient.name}'s Health Overview</h3>
          
          <div className="patient-info-grid">
            <div className="info-card">
              <span className="info-label">Age</span>
              <span className="info-value">{selectedPatient.age}</span>
            </div>
            
            <div className="info-card">
            <span className="info-label">Alzheimer's Stage</span>
              <span className="info-value">{getStageDescription(selectedPatient.alzheimer_stage)}</span>
            </div>
            
            {/* Add more patient details as needed */}
          </div>

          <div className="patient-actions">
            <button 
              className="btn-upload"
              onClick={() => onMRIClick(selectedPatient)}
            >
              <i className="icon-mri"></i>Upload MRI Scan
              </button>
            <button 
              className="btn-upload"
              onClick={() => onImageClick(selectedPatient)}
            >
             <i className="icon-image"></i> Upload Patient Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PatientStats;

