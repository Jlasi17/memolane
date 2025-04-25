import { useState, useEffect } from 'react';
import './styles.css';

const TimeSlotPicker = ({ selectedDate, appointments, onTimeSelect }) => {
  const timeSlots = [];
  const startHour = 9; // 9 AM
  const endHour = 17; // 5 PM
  
  // Generate time slots
  for (let hour = startHour; hour <= endHour; hour++) {
    for (let minute = 0; minute < 60; minute += 30) {
      const time = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;
      const slotTime = new Date(`${selectedDate}T${time}`);
      const isAvailable = !appointments.some(appt => {
        const existingTime = new Date(`${appt.date}T${appt.time}`);
        return Math.abs(slotTime - existingTime) < 30 * 60 * 1000;
      });
      
      timeSlots.push(
        <button
          key={time}
          type="button"
          className={`time-slot ${isAvailable ? 'available' : 'unavailable'}`}
          onClick={() => isAvailable && onTimeSelect(time)}
          disabled={!isAvailable}
        >
          {time}
          {!isAvailable && <span className="slot-tooltip">Booked or too close to another appointment</span>}
        </button>
      );
    }
  }

  return (
    <div className="time-slot-grid">
      {timeSlots}
    </div>
  );
};

const ScheduleAppointment = ({ appointments, onAddAppointment, onCompleteAppointment, apiBaseUrl }) => {
  const [showAddForm, setShowAddForm] = useState(false);
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [formData, setFormData] = useState({
    patient_id: '',
    date: '',
    time: '',
    description: ''
  });
  const [formError, setFormError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  // Fetch patients when component mounts
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
        
        // Set initial patient_id if patients exist
        if (data.patients?.length > 0) {
          setFormData(prev => ({
            ...prev,
            patient_id: data.patients[0].patient_id
          }));
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPatients();
  }, [apiBaseUrl]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    // Clear messages when user makes changes
    setFormError('');
    setSuccessMessage('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Clear previous messages
    setFormError('');
    setSuccessMessage('');
    
    // Validate form
    if (!formData.patient_id || !formData.date || !formData.time) {
      setFormError('Please fill all required fields');
      return;
    }

    try {
      // Check for time conflicts including 30-minute buffer
      const appointmentTime = new Date(`${formData.date}T${formData.time}`);
      const timeBuffer = 30 * 60 * 1000; // 30 minutes in milliseconds
      
      const hasConflict = appointments.some(appt => {
        const existingTime = new Date(`${appt.date}T${appt.time}`);
        return Math.abs(appointmentTime - existingTime) < timeBuffer;
      });

      if (hasConflict) {
        setFormError('Cannot book within 30 minutes of existing appointments');
        return;
      }

      // Call the API
      await onAddAppointment(formData);
      
      // Reset form but keep it open
      setFormData({
        patient_id: patients.length > 0 ? patients[0].patient_id : '',
        date: '',
        time: '',
        description: ''
      });
      
      // Show success message
      setSuccessMessage('Appointment scheduled successfully!');
      
    } catch (error) {
      setFormError(error.message || 'Failed to schedule appointment');
    }
  };

  if (loading) return <div className="loading-message">Loading patient data...</div>;
  if (error) return <div className="error-message">Error: {error}</div>;
  if (patients.length === 0) return <div className="no-patients">No patients found. You need patients to schedule appointments.</div>;

  return (
    <div className="appointment-container">
      <div className="appointment-list">
        <h3>Upcoming Appointments</h3>
        {appointments.length === 0 ? (
          <p>No appointments scheduled</p>
        ) : (
          <ul>
            {appointments.map(appt => (
              <li key={appt.id} className="appointment-item">
                <div>
                  <strong>Patient:</strong> {appt.patient_name || appt.patient_id}<br />
                  <strong>Date:</strong> {appt.date} <strong>Time:</strong> {appt.time}<br />
                  {appt.description && <><strong>Notes:</strong> {appt.description}</>}
                </div>
                <button 
                  onClick={() => onCompleteAppointment(appt.id)}
                  className="complete-btn"
                >
                  Done
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <button 
        onClick={() => {
          setShowAddForm(true);
          setFormError('');
          setSuccessMessage('');
        }}
        className="add-btn"
      >
        Add Appointment
      </button>

      {showAddForm && (
        <div className="add-appointment-form">
          <h4>New Appointment</h4>
          
          {/* Success message */}
          {successMessage && (
            <div className="success-message">
              {successMessage}
            </div>
          )}
          
          {/* Error message */}
          {formError && (
            <div className="error-message">
              {formError}
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Patient</label>
              <select
                name="patient_id"
                value={formData.patient_id}
                onChange={handleInputChange}
                required
              >
                {patients.map(patient => (
                  <option key={patient.patient_id} value={patient.patient_id}>
                    {patient.name} (ID: {patient.patient_id})
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Date</label>
              <input
                type="date"
                name="date"
                value={formData.date}
                onChange={handleInputChange}
                required
                min={new Date().toISOString().split('T')[0]}
              />
            </div>
            
            {formData.date && (
              <div className="form-group">
                <label>Available Time Slots</label>
                <TimeSlotPicker 
                  selectedDate={formData.date}
                  appointments={appointments}
                  onTimeSelect={(time) => {
                    setFormData({...formData, time});
                    setFormError('');
                    setSuccessMessage('');
                  }}
                />
                <input
                  type="time"
                  name="time"
                  value={formData.time}
                  onChange={handleInputChange}
                  required
                  style={{ marginTop: '10px' }}
                />
              </div>
            )}

            <div className="form-group">
              <label>Description (Optional)</label>
              <textarea
                name="description"
                value={formData.description}
                onChange={handleInputChange}
              />
            </div>

            <div className="form-actions">
              <button type="submit" className="submit-btn">Schedule</button>
              <button 
                type="button" 
                onClick={() => {
                  setShowAddForm(false);
                  setFormError('');
                  setSuccessMessage('');
                }}
                className="cancel-btn"
              >
                Close
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};

export default ScheduleAppointment;