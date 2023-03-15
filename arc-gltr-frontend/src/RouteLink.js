import React from 'react';
import { Routes, Route } from 'react-router-dom';

import Main from './Main';
import Status from './Status';

const RouteLink = () => {
  return (
    <Routes> 
      <Route path='/' element={<Main/>}></Route>
      <Route path='/stats' element={<Status/>}></Route>
    </Routes>
  );
}

export default RouteLink;