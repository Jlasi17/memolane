import { useState, useEffect } from 'react';
import './styles.css';

const AddMedication = ({ medications = [], setMedications, onAddMedication, onDeleteMedication, apiBaseUrl }) => {
  const [showAddForm, setShowAddForm] = useState(false);
  const [formData, setFormData] = useState({
    patient_id: '',
    name: '',
    time: [],
    duration: 7,
    notes: ''
  });
  const [error, setError] = useState('');
  const [patients, setPatients] = useState([]);
  const [loadingPatients, setLoadingPatients] = useState(true);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/user_patients`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
    
        const contentType = response.headers.get("content-type");
    
        if (!response.ok) {
          if (contentType && contentType.includes("application/json")) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to fetch patients');
          } else {
            const text = await response.text();
            throw new Error(`Error ${response.status}: ${text}`);
          }
        }
    
        const data = await response.json();
        setPatients(data.patients || []);
    
        if (data.patients?.length > 0) {
          setFormData(prev => ({ ...prev, patient_id: data.patients[0].patient_id }));
        }
      } catch (err) {
        console.error('Fetch error:', err);
        setError(err.message);
      } finally {
        setLoadingPatients(false);
      }
    };

    fetchPatients();
  }, [apiBaseUrl]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleTimeChange = (e) => {
    const { value, checked } = e.target;
    setFormData(prev => {
      if (checked) {
        return { ...prev, time: [...prev.time, value] };
      } else {
        return { ...prev, time: prev.time.filter(t => t !== value) };
      }
    });
  };

  const handleDeleteMedication = async (medicationId) => {
    try {
      if (!medicationId) {
        throw new Error('Invalid medication ID');
      }
  
      console.log('Deleting medication ID:', medicationId);
  
      const response = await fetch(`${apiBaseUrl}/api/medications/${medicationId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        }
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to delete medication');
      }
  
      // Update local state - use the same ID field as in your data
      setMedications(prev => prev.filter(med => med.id !== medicationId));
      
    } catch (err) {
      console.error('Delete error:', err);
      setError(err.message || 'Failed to delete medication');
    }
  };



  const handleSubmit = async (e) => {
    e.preventDefault();
  
    if (!formData.patient_id || !formData.name || formData.time.length === 0) {
      setError('Please fill all required fields');
      return;
    }
  
    setError('');
  
    try {
      const response = await fetch(`${apiBaseUrl}/api/medications`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(formData)
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to add medication');
      }
  
      const newMedication = await response.json();
      setMedications(prev => [...prev, newMedication]);
  
      setShowAddForm(false);
      setFormData(prev => ({
        ...prev,
        name: '',
        time: [],
        duration: 7,
        notes: ''
      }));
    } catch (err) {
      console.error('Add error:', err);
      setError(err.message || 'Failed to add medication');
    }
  };



  useEffect(() => {
  console.log('Current medications:', medications);
}, [medications]);

  if (loadingPatients) return <div>Loading patients...</div>;

  return (
    <div className="medication-container">
      <div className="medication-list">
        <h3>Current Medications</h3>
        {medications.length === 0 ? (
          <p>No medications prescribed</p>
        ) : (
          <ul>
            {medications.map(med => {
          return (
            <li key={med.id} className="medication-item">
              <div>
                <strong>Patient:</strong> {med.patient_name || med.patient_id}<br />
                <strong>Medication:</strong> {med.name}<br />
                <strong>Time:</strong> {Array.isArray(med.time) ? med.time.join(', ') : med.time}<br />
                <strong>Duration:</strong> {med.duration} days<br />
                {med.notes && <><strong>Notes:</strong> {med.notes}</>}
              </div>
              <button 
                onClick={() => {
                  if (med.id) {
                    console.log('Deleting medication with ID:', med.id);
                    handleDeleteMedication(med.id);
                  } else {
                    console.error('Medication ID missing:', med);
                    setError('Cannot delete this medication – ID is missing.');
                  }
                }}
                className="delete-btn"
              >
                Delete
              </button>
            </li>
          );
        })}
          </ul>
        )}
      </div>

      <button 
        onClick={() => setShowAddForm(true)}
        className="add-btn"
      >
        Add Medication
      </button>

      {showAddForm && (
        <div className="add-medication-form">
          <h4>New Medication</h4>
          {error && <div className="error-message">{error}</div>}
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Patient</label>
              {loadingPatients ? (
                <select disabled>
                  <option>Loading patients...</option>
                </select>
              ) : (
                <select
                  name="patient_id"
                  value={formData.patient_id}
                  onChange={handleInputChange}
                  required
                >
                  {patients.length === 0 ? (
                    <option value="">No patients available</option>
                  ) : (
                    <>
                      <option value="">Select a patient</option>
                      {patients.map(patient => (
                        <option key={patient.patient_id} value={patient.patient_id}>
                          {patient.name} (ID: {patient.patient_id})
                        </option>
                      ))}
                    </>
                  )}
                </select>
              )}
            </div>
            <div className="form-group">
              <label>Medication Name</label>
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                required
              />
            </div>
            <div className="form-group">
              <label>Time of Day</label>
              <div className="time-options">
                <label>
                  <input
                    type="checkbox"
                    name="time"
                    value="Morning"
                    checked={formData.time.includes('Morning')}
                    onChange={handleTimeChange}
                  />
                  Morning
                </label>
                <label>
                  <input
                    type="checkbox"
                    name="time"
                    value="Afternoon"
                    checked={formData.time.includes('Afternoon')}
                    onChange={handleTimeChange}
                  />
                  Afternoon
                </label>
                <label>
                  <input
                    type="checkbox"
                    name="time"
                    value="Evening"
                    checked={formData.time.includes('Evening')}
                    onChange={handleTimeChange}
                  />
                  Night
                </label>
              </div>
            </div>
            <div className="form-group">
              <label>Duration (days)</label>
              <input
                type="number"
                name="duration"
                min="1"
                value={formData.duration}
                onChange={handleInputChange}
                required
              />
            </div>
            <div className="form-group">
              <label>Notes (Optional)</label>
              <textarea
                name="notes"
                value={formData.notes}
                onChange={handleInputChange}
              />
            </div>
            <div className="form-actions">
              <button type="submit" className="submit-btn">Prescribe</button>
              <button 
                type="button" 
                onClick={() => setShowAddForm(false)}
                className="cancel-btn"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};

export default AddMedication;