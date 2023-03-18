import React, { useEffect, useState } from 'react';
import IllinoisLOGO from './images/illinois-tech-with-seal.svg'

const Status = () => {
  const getCurrentWeekNumber = () => {
    const now = new Date();
    const onejan = new Date(now.getFullYear(), 0, 1);
    const currentWeek = Math.ceil(((now - onejan) / 86400000 + onejan.getDay() + 1) / 7);
    const currentYear = now.getFullYear();
    return currentWeek +"-"+ currentYear;
  }
  const getCurrentDate = () => {
    const now = new Date();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const year = now.getFullYear();

    return `${month}-${day}-${year}`;
  }
  const getCurrentMonth = () => {
    const now = new Date();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const year = now.getFullYear();

    return `${month}-${year}`;
  }

  const [date, setDate] = useState(0)
  const [week, setWeek] = useState(0)
  const [month, setMonth] = useState(0)
  const [total, setTotal] = useState(0)

  useEffect(() => {
    getData();
  }, [])

  const getData = async ()=>{
    var data = {}
    console.log(getCurrentWeekNumber());
    // await fetch("http://localhost:5001/stats", {
    await fetch("http://ec2-3-145-192-227.us-east-2.compute.amazonaws.com:5001/stats", {
      method: 'GET',
      mode: "cors",
      }).then(response=>{
          data = response;
      });
      data = await data.json();
      setWeek(data["week"][getCurrentWeekNumber()]?data["week"][getCurrentWeekNumber()]:0);
      setDate(data["day"][getCurrentDate()]?data["day"][getCurrentDate()]:0);
      setMonth(data["month"][getCurrentMonth()]?data["month"][getCurrentMonth()]:0);
      let tSum=0
      for (let key in data["month"]) {
        if (data["month"].hasOwnProperty(key)) {
          tSum += data["month"][key];
        }
      }
      setTotal(tSum);
  }

  return (
    <div>
       <div className='container'>
          <div className='card-stats'>
            <h1 className='m-0'>STATS</h1>
            <div style={{display:"flex", justifyContent:"space-between", alignContent:"center", 
            borderTop: "1px solid rgba(123, 123, 123, 0.75)", marginTop: "25px", 
            borderBottom: "1px solid rgba(123, 123, 123, 0.75)"}}>
              <div>
                  <h3 style={{color: "rgba(123, 123, 123, 0.85)", margin: "17px 0px 0px 0px"}}>Today</h3>
                  <p style={{fontSize: "40px", margin: "9px 0px 9px 0px"}}>{date}</p>
              </div>
              <div>
                  <h3 style={{color: "rgba(123, 123, 123, 0.85)", margin: "17px 0px 0px 0px"}}>Week</h3>
                  <p style={{fontSize: "40px", margin: "9px 0px 9px 0px"}}>{week}</p>
              </div>
              <div>
                  <h3 style={{color: "rgba(123, 123, 123, 0.85)", margin: "17px 0px 0px 0px"}}>Month</h3>
                  <p style={{fontSize: "40px", margin: "9px 0px 9px 0px"}}>{month}</p>
              </div>
              <div>
                  <h3 style={{color: "rgba(123, 123, 123, 0.85)", margin: "17px 0px 0px 0px"}}>Total</h3>
                  <p style={{fontSize: "40px", margin: "9px 0px 9px 0px"}}>{total}</p>
              </div>
            </div>
          <div style={{paddingTop:"20px"}}>
              <img src={IllinoisLOGO} alt="illinois tech" width="200px"/>
          </div>
          </div>
       </div>
    </div>
    );
}

export default Status;