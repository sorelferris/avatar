// @ts-ignore
import { makeCylinder, sketchCircle } from 'replicad';

export const params = {
  outerRadius: 75,
  innerRadius: 75 - 15,
  headLength: 100,
  headDepth: 100,
  coverThick: 2,
};

const { outerRadius, innerRadius, headLength, coverThick } = params;

export default () => {
  const outerContour = sketchCircle(outerRadius); // Head outer contour
  const innerContour = sketchCircle(innerRadius); // Head inner contour (for hollowing)

  // Hollow head cylinder with inner contour cut
  let headMain = outerContour.extrude(headLength);
  headMain = headMain.cut(innerContour.extrude(headLength));

  return [{ shape: headMain, color: '#3a3a3a', name: 'hollow_head_cylinder' }];
};
